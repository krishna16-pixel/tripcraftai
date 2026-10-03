"""
live_chat.py
------------
Live-research chat for TripCraft.

Gives the /chat endpoint real, working tools (no `uvx`, no LangChain
AgentExecutor, no extra API keys required):

  web_search         Tavily (if TAVILY_API_KEY set, 10 results) else DuckDuckGo (10)
  fetch_page         read a web page's text
  get_weather        Open-Meteo forecast (free, no key)
  get_exchange_rate  open.er-api.com (free, no key)
  get_current_datetime  real clock for any timezone

`stream_live_chat()` runs a small tool-calling loop on a chat model and
yields dict events the SSE layer relays to the browser:

  {"status": "Searching the web: ...", "detail": "...", "tool": "..."} live progress
  {"sources": [{title,url,domain}]}    fetched-source metadata for the Sources drawer
  {"token": "..."}                       answer text
  {"error": "..."}                       failure

If the model/endpoint rejects tool calling, it falls back to
"search first, then answer" so the user still gets live data.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

logger = logging.getLogger("trip_planner.live_chat")

HTTP_TIMEOUT = 15.0
UA = {"User-Agent": "Mozilla/5.0 (TripCraftAI trip planner)"}
SEARCH_MAX_RESULTS = 10

# --------------------------------------------------------------------------
# Custom user prompt (plain txt -- same file as app.py: my_prompt.txt)
# --------------------------------------------------------------------------
_CUSTOM_PROMPT_CACHE = {"mtime": 0.0, "text": ""}


def get_custom_prompt() -> str:
    path = os.getenv(
        "CUSTOM_PROMPT_FILE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_prompt.txt"),
    )
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""
    if mtime != _CUSTOM_PROMPT_CACHE.get("mtime"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _CUSTOM_PROMPT_CACHE.update({"mtime": mtime, "text": f.read().strip()})
        except Exception as exc:
            logger.warning("Could not read custom prompt %s: %s", path, exc)
            return ""
    return _CUSTOM_PROMPT_CACHE.get("text", "")


def _with_custom_prompt(base: str) -> str:
    custom = get_custom_prompt()
    if not custom:
        return base
    return f"Custom user instructions (highest priority, always follow):\n{custom}\n\n{base}"


# --------------------------------------------------------------------------
# Sources shown in the frontend's Sources drawer.
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s\)\]]+")


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _extract_sources(search_text: str) -> List[dict]:
    """Parse `- title\\n  url\\n  snippet` search output into logo-card sources."""
    sources: List[dict] = []
    if not search_text:
        return sources
    # Structured lines first (title + url pairs)
    lines = search_text.splitlines()
    pending_title = ""
    for line in lines:
        s = line.strip()
        if s.startswith("- ") and len(s) > 2:
            pending_title = s[2:].strip()[:120]
            continue
        m = _URL_RE.search(s)
        if m:
            url = m.group(0).rstrip(".,;")
            domain = _domain_of(url)
            if not domain:
                continue
            title = pending_title or domain
            sources.append({"title": title, "url": url, "domain": domain})
            pending_title = ""
            if len(sources) >= SEARCH_MAX_RESULTS:
                break
    # Fallback: any bare URLs in the text
    if not sources:
        for m in _URL_RE.finditer(search_text):
            url = m.group(0).rstrip(".,;")
            domain = _domain_of(url)
            if domain:
                sources.append({"title": domain, "url": url, "domain": domain})
            if len(sources) >= SEARCH_MAX_RESULTS:
                break
    # Dedupe by URL
    seen = set()
    uniq: List[dict] = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            uniq.append(s)
    return uniq[:SEARCH_MAX_RESULTS]

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def _search_sync(query: str, max_results: int = SEARCH_MAX_RESULTS) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if key:
        try:
            from tavily import TavilyClient

            res = TavilyClient(api_key=key).search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            rows = [
                f"- {r.get('title', '')}\n  {r.get('url', '')}\n  {(r.get('content') or '')[:400]}"
                for r in res.get("results", [])
            ]
            if rows:
                return "\n".join(rows)
        except Exception as exc:
            logger.warning("Tavily failed (%s); trying DuckDuckGo", exc)

    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        rows = [
            f"- {r.get('title', '')}\n  {r.get('href', '')}\n  {(r.get('body') or '')[:400]}"
            for r in results
        ]
        if rows:
            return "\n".join(rows)
        return "No results found. Try a different query."
    except Exception as exc:
        logger.warning("DuckDuckGo failed: %s", exc)
        return f"Search failed ({exc}). Try a different query or say the fact is unverified."


@tool
async def web_search(query: str) -> str:
    """Search the live web (up to 10 websites per query). Use for current prices, opening hours,
    events, visa/entry rules, transport options, news, hotel areas, safety notes.
    Input: a short, specific query."""
    return await asyncio.to_thread(_search_sync, query)


@tool
async def fetch_page(url: str) -> str:
    """Fetch and read the text of a web page (use after web_search when a snippet isn't enough)."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=UA) as client:
            r = await client.get(url)
            r.raise_for_status()
        text = r.text
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        return text[:6000] or "Page had no readable text."
    except Exception as exc:
        return f"Could not fetch {url}: {exc}"


_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "showers",
    81: "showers", 82: "violent showers", 95: "thunderstorm", 96: "thunderstorm w/ hail",
    99: "thunderstorm w/ hail",
}


@tool
async def get_weather(city: str, days: int = 7) -> str:
    """Live weather forecast for a city (up to 16 days ahead). Input: city name, number of days."""
    days = max(1, min(int(days or 7), 16))
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=UA) as client:
            g = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
            )
            g.raise_for_status()
            hits = g.json().get("results") or []
            if not hits:
                return f"Could not find a location called '{city}'."
            place = hits[0]
            f = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            f.raise_for_status()
            d = f.json()["daily"]
        lines = [f"Forecast for {place['name']}, {place.get('country', '')} (local dates, °C):"]
        for i, day in enumerate(d["time"]):
            lines.append(
                f"- {day}: {_WMO.get(d['weathercode'][i], 'n/a')}, "
                f"{d['temperature_2m_min'][i]}–{d['temperature_2m_max'][i]}°C, "
                f"rain chance {d['precipitation_probability_max'][i]}%"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Weather lookup failed: {exc}"


@tool
async def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Live currency exchange rate. Input: 3-letter codes, e.g. USD, AED, INR."""
    a, b = from_currency.upper().strip(), to_currency.upper().strip()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=UA) as client:
            r = await client.get(f"https://open.er-api.com/v6/latest/{a}")
            r.raise_for_status()
            rates = r.json().get("rates", {})
        if b not in rates:
            return f"No rate found for {a}->{b}."
        return f"1 {a} = {rates[b]} {b} (live rate)."
    except Exception as exc:
        return f"Exchange rate lookup failed: {exc}"


@tool
async def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Current real date/time in an IANA timezone, e.g. 'Asia/Dubai', 'Asia/Kolkata'."""
    try:
        now = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now = datetime.now(timezone.utc)
        timezone_name = "UTC"
    return f"{now.strftime('%A, %Y-%m-%d %H:%M')} ({timezone_name})"


TOOLS = [web_search, fetch_page, get_weather, get_exchange_rate, get_current_datetime]
TOOL_MAP = {t.name: t for t in TOOLS}


def _status_for(name: str, args: dict) -> str:
    if name == "web_search":
        return f"Searching the web: {str(args.get('query', ''))[:80]}"
    if name == "fetch_page":
        return f"Reading a web page: {str(args.get('url', ''))[:80]}"
    if name == "get_weather":
        return f"Checking live weather: {args.get('city', '')}"
    if name == "get_exchange_rate":
        return f"Checking exchange rate {args.get('from_currency', '')}→{args.get('to_currency', '')}"
    if name == "get_current_datetime":
        return "Checking the date and time"
    return f"Running {name}"


def _detail_for(name: str, args: dict) -> str:
    """Raw backend detail shown when the thinking label is clicked (Claude-style)."""
    try:
        return json.dumps({k: str(v)[:200] for k, v in (args or {}).items()}, ensure_ascii=False)
    except Exception:
        return str(args)[:300]


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

_TRAVEL_RE = re.compile(
    r"\b(trip|travel\w*|itinerar\w*|flights?|hotels?|hostels?|visa|weather|forecast|"
    r"tomorrow|tonight|weekend|price\w*|cost\w*|budget|open(ing)?|hours|tickets?|book(ing)?|"
    r"visit\w*|tour\w*|things to do|restaurants?|currency|exchange|vacation|holiday|"
    r"stay|route|train|metro|plan|dubai|latest|current|today|now|news|event\w*)\b",
    re.I,
)


def needs_live_research(message: str) -> bool:
    """Use live tools only when this turn asks for travel/current information.

    Do not let an earlier travel question force later small talk onto the
    slower research path; the conversation history is still passed to either
    model for context.
    """
    return bool(_TRAVEL_RE.search(message))


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------


def build_system_prompt() -> str:
    now = datetime.now(timezone.utc)
    base = (
        "You are TripCraft, a friendly, practical travel-planning assistant.\n"
        f"Current date/time: {now.strftime('%A, %Y-%m-%d %H:%M')} UTC. Resolve 'today', "
        "'tomorrow', 'this weekend' from this date (use get_current_datetime for a "
        "destination's local date/time).\n\n"
        "You HAVE live tools: web_search, fetch_page, get_weather, get_exchange_rate, "
        "get_current_datetime. For anything time-sensitive or factual about a destination "
        "(weather, prices, opening hours, events, visa/entry rules, transport, safety), CALL "
        "THE TOOLS first -- never guess and never say you cannot check live info. Make several "
        "targeted calls (weather + prices + attractions/hours + transport/visa) before "
        "answering a planning request. web_search returns up to 10 websites per query -- "
        "use 2-3 targeted queries for good coverage. Do not write any text before your tool calls.\n\n"
        "When asked to plan a trip: if dates/budget are missing, assume sensible defaults, state "
        "them in one line, and still deliver the full plan. Give a day-by-day itinerary with "
        "times, estimated costs (with the local currency and a conversion if useful), the live "
        "weather, transport tips, and a short 'check before you go' list (visa, bookings). "
        "Only state as verified what the tools "
        "returned; label anything else as an estimate. For casual chat, reply briefly.\n\n"
        "SOURCES: fetched websites are available in the UI's Sources side panel. Do not add "
        "a source list, raw source URLs, citation numbers, or markdown footnotes to the answer "
        "unless the user explicitly asks for source links. Refer to a source by name only when "
        "it materially helps.\n\n"
        "FORMAT: make substantial answers easy to scan. Lead with a brief direct answer, "
        "then use descriptive Markdown headings and concise bullets or numbered steps. "
        "Use a Markdown table for side-by-side options, prices, durations, or pros/cons when "
        "that is clearer than prose. Keep casual replies short, avoid repetitive headings, "
        "and never emit raw HTML."
    )
    return _with_custom_prompt(base)


def _to_lc_messages(history: List[dict], message: str) -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=build_system_prompt())]
    for turn in history[-10:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        msgs.append(cls(content=turn["content"]))
    msgs.append(HumanMessage(content=message))
    return msgs


def _chunk_text(chunk) -> str:
    c = chunk.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


async def _run_tool(call: dict) -> str:
    name, args = call["name"], call.get("args") or {}
    t = TOOL_MAP.get(name)
    if t is None:
        return f"Unknown tool {name}."
    try:
        out = await t.ainvoke(args)
        return str(out)[:7000]
    except Exception as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return f"Tool {name} failed: {exc}"


async def stream_live_chat(
    agent_llm,
    message: str,
    history: List[dict],
    max_steps: int = 6,
) -> AsyncIterator[Dict[str, str]]:
    """Tool-calling loop. `agent_llm` is a streaming ChatOpenAI-style model."""
    messages = _to_lc_messages(history, message)
    answered = False
    seen_sources: Dict[str, dict] = {}

    def _sources_event() -> Optional[dict]:
        if not seen_sources:
            return None
        return {"sources": list(seen_sources.values())[:SEARCH_MAX_RESULTS]}

    def _ingest_tool_sources(call: dict, result: str) -> None:
        name = call.get("name", "")
        args = call.get("args") or {}
        if name == "web_search":
            for s in _extract_sources(result or ""):
                seen_sources.setdefault(s["url"], s)
        elif name == "fetch_page":
            url = str(args.get("url", "")).strip()
            if url.startswith("http"):
                domain = _domain_of(url)
                if domain:
                    seen_sources.setdefault(url, {"title": domain, "url": url, "domain": domain})

    try:
        llm_tools = agent_llm.bind_tools(TOOLS)
        for step in range(max_steps):
            final_step = step == max_steps - 1
            runner = agent_llm if final_step else llm_tools
            if final_step:
                messages.append(HumanMessage(content="Now write the final answer using what you found. Do not add a source list, source URLs, citation numbers, or footnotes; fetched sources are available through the UI's Sources side panel."))
            yield {
                "status": "Thinking" if step == 0 else "Putting it together",
                "detail": f"step {step + 1}/{max_steps} reasoning with {agent_llm.model_name if hasattr(agent_llm, 'model_name') else 'GLM'}",
                "tool": "reasoning",
            }

            gathered = None
            async for chunk in runner.astream(messages):
                gathered = chunk if gathered is None else gathered + chunk
                text = _chunk_text(chunk)
                if text:
                    answered = True
                    yield {"token": text}

            calls = list(getattr(gathered, "tool_calls", None) or []) if gathered is not None else []
            if not calls or final_step:
                if not answered:
                    yield {"error": "The AI returned an empty response. Please try again."}
                else:
                    ev = _sources_event()
                    if ev:
                        yield ev
                return

            messages.append(gathered)
            if answered:
                yield {"token": "\n\n"}
            for call in calls:
                args = call.get("args") or {}
                yield {
                    "status": _status_for(call["name"], args),
                    "detail": _detail_for(call["name"], args),
                    "tool": call["name"],
                }
            results = await asyncio.gather(*[_run_tool(c) for c in calls])
            for call, result in zip(calls, results):
                _ingest_tool_sources(call, result)
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
            ev = _sources_event()
            if ev:
                yield ev
        return
    except Exception as exc:
        if answered:
            logger.exception("Live chat failed mid-answer")
            yield {"error": "The AI request failed part-way. Please try again."}
            return
        logger.warning("Tool-calling path failed (%s); using search-first fallback", exc)

    # ---- Fallback: pre-fetch live data, then a plain streamed answer ----
    try:
        yield {"status": "Searching the web", "detail": message[:200], "tool": "web_search"}
        queries = await asyncio.gather(
            _run_tool({"name": "web_search", "args": {"query": message[:200]}}),
            _run_tool({"name": "web_search", "args": {"query": f"{message[:150]} prices opening hours travel tips"}}),
        )
        for q in queries:
            for s in _extract_sources(q or ""):
                seen_sources.setdefault(s["url"], s)
        ev = _sources_event()
        if ev:
            yield ev
        context = "\n\n".join(f"Search results {i + 1}:\n{q}" for i, q in enumerate(queries))
        fb = _to_lc_messages(history, message)
        fb[0] = SystemMessage(content=fb[0].content + "\n\nLIVE SEARCH RESULTS (fetched just now):\n" + context)
        yield {"status": "Writing your answer", "detail": "composing final answer from live results", "tool": "reasoning"}
        got = False
        async for chunk in agent_llm.astream(fb):
            text = _chunk_text(chunk)
            if text:
                got = True
                yield {"token": text}
        if not got:
            yield {"error": "The AI returned an empty response. Please try again."}
        else:
            ev = _sources_event()
            if ev:
                yield ev
    except Exception:
        logger.exception("Fallback path failed")
        yield {"error": "The AI request failed. Check NVIDIA_API_KEY and the model configuration in Render."}
