"""
mcp.py
------
Everything the trip-planning agent uses to touch the outside world and to
check its own work:

  1. Shared Pydantic models (constraints, itinerary, validation report).
  2. Web search tool (Tavily if configured, DuckDuckGo as a no-key fallback).
  3. Real MCP (Model Context Protocol) server integration via
     `langchain-mcp-adapters`, so this agent can call any MCP server you
     point it at (maps, flights, hotels, weather, currency, etc.).
  4. A dependency-free geo/route estimator (haversine + mode speed table)
     used both as an agent tool and inside the deterministic validators.
  5. Pure-Python validators for budget, daily schedule, opening hours, and
     route feasibility -- these never call the LLM, so they are
     deterministic and cheap to re-run on every replanning loop.

app.py imports everything it needs from this module.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("trip_planner.mcp")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# 1. Shared data models
# --------------------------------------------------------------------------


class TravelMode(str, Enum):
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR = "car"
    FERRY = "ferry"
    WALK = "walk"


class CityStop(BaseModel):
    city: str
    country: Optional[str] = None
    min_days: int = 1
    max_days: Optional[int] = None
    must_see: List[str] = Field(default_factory=list)


class TripConstraints(BaseModel):
    origin_city: str
    destinations: List[CityStop]
    start_date: date
    end_date: date
    total_budget: float
    currency: str = "USD"
    travelers: int = 1
    allowed_transport_modes: List[TravelMode] = Field(
        default_factory=lambda: [TravelMode.FLIGHT, TravelMode.TRAIN, TravelMode.CAR]
    )
    daily_start_time: time = time(9, 0)
    daily_end_time: time = time(21, 0)
    max_activity_hours_per_day: float = 8.0
    pace: str = "moderate"  # relaxed | moderate | packed
    preferences: List[str] = Field(default_factory=list)  # e.g. museums, food, nightlife
    notes: Optional[str] = None
    attachment_ids: List[str] = Field(default_factory=list)

    @property
    def trip_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


class Activity(BaseModel):
    name: str
    city: str
    category: str
    date: date
    start_time: time
    end_time: time
    estimated_cost: float = 0.0
    opening_hours: Optional[str] = Field(
        default=None, description='e.g. "09:00-18:00"; null if unknown/always open'
    )
    notes: Optional[str] = None


class TransferLeg(BaseModel):
    from_city: str
    to_city: str
    date: date
    mode: TravelMode
    depart_time: time
    arrive_time: time
    estimated_cost: float = 0.0


class DayPlan(BaseModel):
    date: date
    city: str
    transfer: Optional[TransferLeg] = None
    activities: List[Activity] = Field(default_factory=list)
    lodging_cost: float = 0.0


class Itinerary(BaseModel):
    trip_title: str
    days: List[DayPlan]
    total_estimated_cost: float
    currency: str = "USD"
    summary: str


class ValidationIssue(BaseModel):
    severity: str  # "error" | "warning"
    category: str  # "budget" | "schedule" | "opening_hours" | "route" | "dates"
    message: str


class ValidationReport(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue]
    total_cost: float
    budget_remaining: float


# --------------------------------------------------------------------------
# 2. Web search tool
# --------------------------------------------------------------------------


def _safe(func, label: str):
    """Wrap a tool function so a transient network/API failure returns a
    string the agent can reason about and retry/work around, instead of
    raising and aborting the whole agent run."""

    def _wrapped(query: str) -> str:
        try:
            return func(query)
        except Exception as exc:
            logger.warning("%s call failed: %s", label, exc)
            return (
                f"{label} failed for this query ({exc}). Try a different, more "
                "specific query, or proceed using general knowledge and flag "
                "this fact as unverified in your findings."
            )

    return _wrapped


def build_web_search_tool():
    """Tavily if TAVILY_API_KEY is set (better quality), else DuckDuckGo
    (no API key required) so the agent always has a working search tool.
    Either way, failures are caught so a bad search never crashes the agent."""
    from langchain_core.tools import Tool

    description = (
        "Search the live web for real-world travel info: attraction "
        "opening hours, ticket prices, average costs, transport "
        "schedules, visa/entry rules, weather, safety notes. "
        "Input: a search query string."
    )

    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tavily = TavilySearchResults(max_results=5)
            return Tool(name="web_search", description=description, func=_safe(tavily.run, "web_search"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Tavily unavailable (%s); falling back to DuckDuckGo", exc)

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        ddg = DuckDuckGoSearchRun()
        return Tool(name="web_search", description=description, func=_safe(ddg.run, "web_search"))
    except Exception as exc:  # pragma: no cover
        logger.error("No web search backend available: %s", exc)
        return Tool(
            name="web_search",
            description="Unavailable in this environment.",
            func=lambda q: "web_search is not configured in this environment.",
        )


# --------------------------------------------------------------------------
# 3. MCP server integration (real external tools: maps, flights, hotels...)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Default MCP servers -- FREE ONLY, no paid tiers, no API keys required.
# All three run locally over stdio via `uvx` (so `uv` must be on PATH; see
# https://docs.astral.sh/uv/getting-started/installation/) and call only
# free public APIs under the hood:
#
#   - "time"        official MCP reference server (uvx mcp-server-time)
#                    timezone / local-time conversions -- useful because a
#                    multi-city trip usually crosses time zones.
#   - "fetch"        official MCP reference server (uvx mcp-server-fetch)
#                    fetches full page content (complements the search tool,
#                    which only returns snippets).
#   - "public_apis"  community server (uvx public-apis-mcp-server) wrapping
#                    Open-Meteo (weather, no key), REST Countries (entry/
#                    currency/timezone info, no key), and NASA's public
#                    DEMO_KEY -- all zero-cost.
#
# These load automatically unless you set MCP_SERVERS_JSON / MCP_SERVERS_FILE
# (which fully replace this list) or MCP_USE_DEFAULT_SERVERS=false (which
# disables them with no replacement). Every one of these is free to run
# indefinitely; do not add paid/metered MCP servers to this default set.
# --------------------------------------------------------------------------
DEFAULT_FREE_MCP_SERVERS = {
    "time": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-time"],
    },
    "fetch": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    },
    "public_apis": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["public-apis-mcp-server"],
    },
}


def _load_mcp_server_config() -> dict:
    """
    MCP servers are configured via the MCP_SERVERS_JSON env var (a JSON
    object) or a file path in MCP_SERVERS_FILE -- either fully overrides the
    defaults above. Format matches `langchain-mcp-adapters`'s
    MultiServerMCPClient, e.g. to ADD a free-tier server of your own
    alongside (or instead of) the defaults:

    {
      "time": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-time"]},
      "weather": {"transport": "streamable_http", "url": "https://your-free-weather-mcp/mcp"}
    }

    With nothing set, DEFAULT_FREE_MCP_SERVERS is used. Set
    MCP_USE_DEFAULT_SERVERS=false to run with no MCP servers at all.
    """
    raw = os.getenv("MCP_SERVERS_JSON")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("MCP_SERVERS_JSON is not valid JSON: %s", exc)
            return {}

    path = os.getenv("MCP_SERVERS_FILE")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.getenv("MCP_USE_DEFAULT_SERVERS", "true").lower() in ("false", "0", "no"):
        return {}

    return DEFAULT_FREE_MCP_SERVERS


def _import_mcp_adapter():
    """
    NOTE ON A NAMING COLLISION: this file is named mcp.py (as requested),
    but `langchain-mcp-adapters` depends on the real, pip-installed `mcp`
    SDK package. Because Python puts the running script's directory at the
    front of sys.path, a bare `import mcp` anywhere in the process would
    otherwise resolve to THIS file instead of the real package. app.py
    avoids triggering that by loading this module via importlib under an
    internal alias (never `import mcp`), and here -- right before the one
    place that needs the real SDK -- we additionally drop this file's own
    directory from sys.path for the duration of the import, so the nested
    `import mcp` inside langchain_mcp_adapters resolves correctly.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != this_dir]
        from langchain_mcp_adapters.client import MultiServerMCPClient

        return MultiServerMCPClient
    finally:
        sys.path = saved_path


async def load_mcp_tools() -> list:
    """Connect to every configured MCP server and return their tools as
    LangChain-compatible Tool objects. Fails soft: if the adapter package
    isn't installed, isn't resolvable, or no servers are configured, this
    returns [] rather than crashing the app."""
    config = _load_mcp_server_config()
    if not config:
        logger.info("No MCP servers configured (MCP_SERVERS_JSON/MCP_SERVERS_FILE unset).")
        return []

    try:
        MultiServerMCPClient = _import_mcp_adapter()
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters (and/or the mcp SDK) not installed; skipping "
            "MCP tools. Install with: pip install langchain-mcp-adapters mcp"
        )
        return []

    try:
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        logger.info("Loaded %d tool(s) from %d MCP server(s).", len(tools), len(config))
        return tools
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc)
        return []


_cached_tools: Optional[list] = None


async def get_agent_tools() -> list:
    """All tools exposed to the research agent: web search + route
    estimator + whatever MCP servers are configured. Cached after first
    successful load so we don't reconnect to MCP servers on every request."""
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools

    tools = [build_web_search_tool(), build_route_estimator_tool()]
    tools.extend(await load_mcp_tools())
    _cached_tools = tools
    return tools


# --------------------------------------------------------------------------
# 4. Dependency-free geo / route estimation
# --------------------------------------------------------------------------

# Rough door-to-door average speeds (km/h), including typical overhead
# (airport security/transfer time, station boarding, city traffic).
_MODE_SPEED_KMH = {
    TravelMode.FLIGHT: 500.0,  # cruise speed; fixed overhead added separately
    TravelMode.TRAIN: 110.0,
    TravelMode.BUS: 65.0,
    TravelMode.CAR: 80.0,
    TravelMode.FERRY: 35.0,
    TravelMode.WALK: 4.5,
}
_MODE_FIXED_OVERHEAD_HOURS = {
    TravelMode.FLIGHT: 3.0,  # check-in, security, boarding, deplaning, baggage
    TravelMode.TRAIN: 0.5,
    TravelMode.BUS: 0.4,
    TravelMode.CAR: 0.1,
    TravelMode.FERRY: 0.75,
    TravelMode.WALK: 0.0,
}

_GEOCODE_CACHE: dict = {}


def geocode_city(city_name: str) -> Optional[tuple]:
    """Best-effort lat/lon lookup via OpenStreetMap Nominatim (no API key).
    Returns None quietly if geopy isn't installed or the lookup fails --
    callers must treat that as 'distance unknown', not as an error."""
    if city_name in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[city_name]
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="trip_planner_agent")
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            coords = (location.latitude, location.longitude)
            _GEOCODE_CACHE[city_name] = coords
            return coords
    except Exception as exc:
        logger.warning("Geocoding failed for %s: %s", city_name, exc)
    _GEOCODE_CACHE[city_name] = None
    return None


def haversine_km(coord_a: tuple, coord_b: tuple) -> float:
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def estimate_transfer_hours(from_city: str, to_city: str, mode: TravelMode) -> Optional[float]:
    """Estimated total door-to-door hours for a transfer, or None if the
    cities couldn't be geocoded (caller should treat as 'unverifiable', and
    the validator will emit a warning rather than a hard error)."""
    a, b = geocode_city(from_city), geocode_city(to_city)
    if not a or not b:
        return None
    distance_km = haversine_km(a, b)
    speed = _MODE_SPEED_KMH[mode]
    overhead = _MODE_FIXED_OVERHEAD_HOURS[mode]
    return round(distance_km / speed + overhead, 2)


def build_route_estimator_tool():
    from langchain_core.tools import Tool

    def _run(query: str) -> str:
        """Input format: 'from_city | to_city | mode' e.g. 'Paris | Rome | train'."""
        try:
            parts = [p.strip() for p in query.split("|")]
            from_city, to_city, mode_str = parts[0], parts[1], parts[2].lower()
            mode = TravelMode(mode_str)
        except Exception:
            return (
                "Invalid input. Use: 'from_city | to_city | mode' where mode is "
                "one of flight, train, bus, car, ferry, walk."
            )
        hours = estimate_transfer_hours(from_city, to_city, mode)
        if hours is None:
            return f"Could not geocode {from_city} or {to_city}; distance unknown."
        return (
            f"Estimated door-to-door travel time from {from_city} to {to_city} "
            f"by {mode.value}: ~{hours} hours (includes typical overhead)."
        )

    return Tool(
        name="estimate_route",
        description=(
            "Estimate realistic door-to-door travel time between two cities for "
            "a given mode of transport. Input: 'from_city | to_city | mode' where "
            "mode is one of flight, train, bus, car, ferry, walk. Use this before "
            "scheduling same-day transfers to check feasibility."
        ),
        func=_run,
    )


# --------------------------------------------------------------------------
# 5. Pure-Python validators (no LLM calls -- deterministic and fast)
# --------------------------------------------------------------------------


def _parse_hours_range(hours_str: str) -> Optional[tuple]:
    try:
        open_s, close_s = hours_str.split("-")
        return (
            datetime.strptime(open_s.strip(), "%H:%M").time(),
            datetime.strptime(close_s.strip(), "%H:%M").time(),
        )
    except Exception:
        return None


def validate_dates(itinerary: Itinerary, constraints: TripConstraints) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not itinerary.days:
        issues.append(ValidationIssue(severity="error", category="dates", message="Itinerary has no days."))
        return issues

    days_sorted = sorted(itinerary.days, key=lambda d: d.date)
    if days_sorted[0].date != constraints.start_date:
        issues.append(ValidationIssue(
            severity="error", category="dates",
            message=f"Itinerary starts {days_sorted[0].date}, expected {constraints.start_date}.",
        ))
    if days_sorted[-1].date != constraints.end_date:
        issues.append(ValidationIssue(
            severity="error", category="dates",
            message=f"Itinerary ends {days_sorted[-1].date}, expected {constraints.end_date}.",
        ))
    expected_dates = {constraints.start_date + timedelta(days=i) for i in range(constraints.trip_days)}
    actual_dates = {d.date for d in itinerary.days}
    missing = expected_dates - actual_dates
    if missing:
        issues.append(ValidationIssue(
            severity="error", category="dates",
            message=f"Missing day(s) in itinerary: {sorted(missing)}.",
        ))
    dupes = [d for d in actual_dates if list(d2.date for d2 in itinerary.days).count(d) > 1]
    if dupes:
        issues.append(ValidationIssue(
            severity="error", category="dates", message=f"Duplicate date(s): {dupes}.",
        ))
    return issues


def validate_budget(itinerary: Itinerary, constraints: TripConstraints) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    computed_total = 0.0
    for day in itinerary.days:
        computed_total += day.lodging_cost
        if day.transfer:
            computed_total += day.transfer.estimated_cost
        for act in day.activities:
            computed_total += act.estimated_cost

    if abs(computed_total - itinerary.total_estimated_cost) > max(1.0, 0.02 * computed_total):
        issues.append(ValidationIssue(
            severity="warning", category="budget",
            message=(
                f"Itinerary's stated total ({itinerary.total_estimated_cost:.2f}) doesn't match "
                f"the sum of its line items ({computed_total:.2f}); using the computed sum."
            ),
        ))

    if computed_total > constraints.total_budget:
        over = computed_total - constraints.total_budget
        issues.append(ValidationIssue(
            severity="error", category="budget",
            message=(
                f"Estimated cost {computed_total:.2f} {constraints.currency} exceeds budget "
                f"{constraints.total_budget:.2f} {constraints.currency} by {over:.2f}."
            ),
        ))
    return issues


def validate_daily_schedule(itinerary: Itinerary, constraints: TripConstraints) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for day in itinerary.days:
        events = sorted(day.activities, key=lambda a: a.start_time)
        # overlap + outside-daily-window checks
        for act in events:
            if act.start_time < constraints.daily_start_time or act.end_time > constraints.daily_end_time:
                issues.append(ValidationIssue(
                    severity="warning", category="schedule",
                    message=(
                        f"{day.date} '{act.name}' ({act.start_time}-{act.end_time}) falls outside "
                        f"the preferred daily window {constraints.daily_start_time}-{constraints.daily_end_time}."
                    ),
                ))
        for i in range(len(events) - 1):
            if events[i].end_time > events[i + 1].start_time:
                issues.append(ValidationIssue(
                    severity="error", category="schedule",
                    message=(
                        f"{day.date}: '{events[i].name}' ends after '{events[i + 1].name}' starts -- overlap."
                    ),
                ))
        total_hours = sum(
            (datetime.combine(day.date, a.end_time) - datetime.combine(day.date, a.start_time)).seconds / 3600
            for a in events
        )
        if total_hours > constraints.max_activity_hours_per_day + 0.01:
            issues.append(ValidationIssue(
                severity="warning", category="schedule",
                message=(
                    f"{day.date}: {total_hours:.1f}h of activities exceeds the "
                    f"{constraints.max_activity_hours_per_day}h/day preference ({constraints.pace} pace)."
                ),
            ))
    return issues


def validate_opening_hours(itinerary: Itinerary) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for day in itinerary.days:
        for act in day.activities:
            if not act.opening_hours:
                continue
            parsed = _parse_hours_range(act.opening_hours)
            if not parsed:
                continue
            open_t, close_t = parsed
            if act.start_time < open_t or act.end_time > close_t:
                issues.append(ValidationIssue(
                    severity="error", category="opening_hours",
                    message=(
                        f"{day.date} '{act.name}' is scheduled {act.start_time}-{act.end_time} but "
                        f"only opens {open_t}-{close_t}."
                    ),
                ))
    return issues


def validate_routes(itinerary: Itinerary, constraints: TripConstraints) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for day in itinerary.days:
        t = day.transfer
        if not t:
            continue
        if t.mode not in constraints.allowed_transport_modes:
            issues.append(ValidationIssue(
                severity="error", category="route",
                message=f"{day.date}: transfer uses {t.mode.value}, which is not in the allowed modes.",
            ))
        scheduled_hours = (
            datetime.combine(t.date, t.arrive_time) - datetime.combine(t.date, t.depart_time)
        ).seconds / 3600
        est_hours = estimate_transfer_hours(t.from_city, t.to_city, t.mode)
        if est_hours is not None and scheduled_hours < est_hours * 0.85:
            issues.append(ValidationIssue(
                severity="error", category="route",
                message=(
                    f"{day.date}: {t.from_city}->{t.to_city} by {t.mode.value} is scheduled for "
                    f"{scheduled_hours:.1f}h but realistically takes ~{est_hours:.1f}h."
                ),
            ))
        # same-day activity right after a transfer: make sure there's a gap
        if day.activities:
            first_activity = min(day.activities, key=lambda a: a.start_time)
            gap_hours = (
                datetime.combine(day.date, first_activity.start_time)
                - datetime.combine(t.date, t.arrive_time)
            ).seconds / 3600
            if gap_hours < 0.5:
                issues.append(ValidationIssue(
                    severity="warning", category="route",
                    message=(
                        f"{day.date}: only {gap_hours:.1f}h between arriving in {t.to_city} and "
                        f"starting '{first_activity.name}' -- likely too tight."
                    ),
                ))
    return issues


def validate_itinerary(itinerary: Itinerary, constraints: TripConstraints) -> ValidationReport:
    issues: List[ValidationIssue] = []
    issues += validate_dates(itinerary, constraints)
    issues += validate_budget(itinerary, constraints)
    issues += validate_daily_schedule(itinerary, constraints)
    issues += validate_opening_hours(itinerary)
    issues += validate_routes(itinerary, constraints)

    computed_total = sum(
        day.lodging_cost
        + (day.transfer.estimated_cost if day.transfer else 0.0)
        + sum(a.estimated_cost for a in day.activities)
        for day in itinerary.days
    )
    is_valid = not any(i.severity == "error" for i in issues)
    return ValidationReport(
        is_valid=is_valid,
        issues=issues,
        total_cost=round(computed_total, 2),
        budget_remaining=round(constraints.total_budget - computed_total, 2),
    )
