"""
live_chat.py
------------
Live-research chat for TripCraft.

Gives the /chat endpoint real, working tools (no `uvx`, no LangChain
AgentExecutor, no extra API keys required):

  web_search         Tavily (if TAVILY_API_KEY set) else DuckDuckGo (ddgs)
  fetch_page         read a web page's text
  get_weather        Open-Meteo forecast (free, no key)
  get_exchange_rate  open.er-api.com (free, no key)
  get_current_datetime  real clock for any timezone

`stream_live_chat()` runs a small tool-calling loop on a chat model and
yields dict events the SSE layer relays to the browser:

  {"status": "Searching the web: ..."}   live progress
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

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def _search_sync(query: str, max_results: int = 6) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if key:
        try:
            from tavily import TavilyClient

            res = TavilyClient(api_key=key).search(query=query, max_results=max_results)
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
    """Search the live web. Use for current prices, opening hours, events, visa/entry rules,
    transport options, news, hotel areas, safety notes. Input: a short, specific query."""
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
        return "Reading a web page"
    if name == "get_weather":
        return f"Checking live weather: {args.get('city', '')}"
    if name == "get_exchange_rate":
        return f"Checking exchange rate {args.get('from_currency', '')}→{args.get('to_currency', '')}"
    if name == "get_current_datetime":
        return "Checking the date and time"
    return f"Running {name}"


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
    return (
        "You are TripCraft, a friendly, practical travel-planning assistant.\n"
        f"Current date/time: {now.strftime('%A, %Y-%m-%d %H:%M')} UTC. Resolve 'today', "
        "'tomorrow', 'this weekend' from this date (use get_current_datetime for a "
        "destination's local date/time).\n\n"
        "You HAVE live tools: web_search, fetch_page, get_weather, get_exchange_rate, "
        "get_current_datetime. For anything time-sensitive or factual about a destination "
        "(weather, prices, opening hours, events, visa/entry rules, transport, safety), CALL "
        "THE TOOLS first -- never guess and never say you cannot check live info. Make several "
        "targeted calls (weather + prices + attractions/hours + transport/visa) before "
        "answering a planning request. Do not write any text before your tool calls.\n\n"
        "When asked to plan a trip: if dates/budget are missing, assume sensible defaults, state "
        "them in one line, and still deliver the full plan. Give a day-by-day itinerary with "
        "times, estimated costs (with the local currency and a conversion if useful), the live "
        "weather, transport tips, and a short 'check before you go' list (visa, bookings). "
        "Mention the source name for important facts. Only state as verified what the tools "
        "returned; label anything else as an estimate. For casual chat, reply briefly."
    )


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

    try:
        llm_tools = agent_llm.bind_tools(TOOLS)
        for step in range(max_steps):
            final_step = step == max_steps - 1
            runner = agent_llm if final_step else llm_tools
            if final_step:
                messages.append(HumanMessage(content="Now write the final answer using what you found."))
            yield {"status": "Thinking" if step == 0 else "Putting it together"}

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
                return

            messages.append(gathered)
            if answered:
                yield {"token": "\n\n"}
            for call in calls:
                yield {"status": _status_for(call["name"], call.get("args") or {})}
            results = await asyncio.gather(*[_run_tool(c) for c in calls])
            for call, result in zip(calls, results):
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
        return
    except Exception as exc:
        if answered:
            logger.exception("Live chat failed mid-answer")
            yield {"error": "The AI request failed part-way. Please try again."}
            return
        logger.warning("Tool-calling path failed (%s); using search-first fallback", exc)

    # ---- Fallback: pre-fetch live data, then a plain streamed answer ----
    try:
        yield {"status": "Searching the web"}
        queries = await asyncio.gather(
            _run_tool({"name": "web_search", "args": {"query": message[:200]}}),
            _run_tool({"name": "web_search", "args": {"query": f"{message[:150]} prices opening hours travel tips"}}),
        )
        context = "\n\n".join(f"Search results {i + 1}:\n{q}" for i, q in enumerate(queries))
        fb = _to_lc_messages(history, message)
        fb[0] = SystemMessage(content=fb[0].content + "\n\nLIVE SEARCH RESULTS (fetched just now):\n" + context)
        yield {"status": "Writing your answer"}
        got = False
        async for chunk in agent_llm.astream(fb):
            text = _chunk_text(chunk)
            if text:
                got = True
                yield {"token": text}
        if not got:
            yield {"error": "The AI returned an empty response. Please try again."}
    except Exception:
        logger.exception("Fallback path failed")
        yield {"error": "The AI request failed. Check NVIDIA_API_KEY and the model configuration in Render."}
