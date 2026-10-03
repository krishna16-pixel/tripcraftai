"""
app.py
------
FastAPI backend for the multi-city trip-planning agent.

Install:
    pip install fastapi uvicorn python-multipart python-dotenv pydantic \
                langchain langchain-openai langchain-community \
                langchain-mcp-adapters mcp geopy duckduckgo-search pypdf
    # also needed for the default free MCP servers (stdio, via uvx):
    #   curl -LsSf https://astral.sh/uv/install.sh | sh
    #   uvx mcp-server-time --help   (first run downloads it)

Run:
    export NVIDIA_API_KEY="nvapi-..."     # free key: https://build.nvidia.com
    uvicorn app:app --reload --port 8000

Env vars:
    NVIDIA_API_KEY        required. Free API key from build.nvidia.com (NVIDIA
                           NIM). GLM-5.3 and GLM-5.3-Flash are both listed
                           there at $0 input/output as of this writing --
                           check build.nvidia.com/z-ai/glm-5-3 for current terms.
    GLM_BASE_URL          default: https://integrate.api.nvidia.com/v1
    GLM_PLANNING_MODEL    default: z-ai/glm-5.3        (planning / reasoning / tool use)
    GLM_MODEL             legacy alias for GLM_PLANNING_MODEL
    GLM_AGENT_MODEL       default: GLM_PLANNING_MODEL (live-research chat with tools)
    GLM_AGENT_MAX_TOKENS  default: 16384 (tool-assisted travel answers)
    GLM_AGENT_TIMEOUT_SECONDS default: 180
    GLM_AGENT_REASONING_EFFORT default: low
    GLM_PLANNING_MAX_TOKENS default: 16384 (structured itinerary drafts)
    RESEARCH_MAX_ITERATIONS default: 5 (research-agent tool steps per pass)
    RESEARCH_TIMEOUT_SECONDS default: 150 (fail fast instead of hanging)
    TRIP_MAX_ITERATIONS   default: 2 (passes for trip planning started from chat)
    TRIP_CHAT_FAST        default: 1 (skip web research + use fast model for chat trips)
    TRIP_FAST_TIMEOUT_SECONDS default: 120
    GEOCODE_TIMEOUT_SECONDS default: 5 (Nominatim lookup cap per city)
    GLM_VISION_MODEL      default: z-ai/glm-5.3-flash  (multimodal -- used for image uploads)
    TAVILY_API_KEY        optional. Better web search than the DuckDuckGo fallback.
                           When set, 10 results per query (search_depth=advanced).
    CUSTOM_PROMPT_FILE    default: ./my_prompt.txt (plain txt you edit; prepended
                           to every AI prompt -- chat, research, draft, intake).
    MCP_SERVERS_JSON      optional. JSON config overriding the default free MCP
                           servers (see mcp.py -- time / fetch / public_apis, all
                           free, no key required).
    MCP_SERVERS_FILE      optional. Path to a JSON file with the same config.
    MCP_USE_DEFAULT_SERVERS  default: true. Set "false" to run with no MCP
                           servers instead of the free defaults.
    MCP_CONNECT_TIMEOUT_SECONDS default: 30 (max wait for MCP servers to connect).
    GLM_CHAT_MAX_TOKENS   default: 16384 (full output budget for chat + citations)
    UPLOAD_DIR             default: ./uploads

Why NVIDIA + GLM-5.3: NVIDIA's API catalog (build.nvidia.com) hosts GLM-5.3
and GLM-5.3-Flash behind a single free NVIDIA_API_KEY, via an OpenAI-
compatible endpoint -- so we talk to it with langchain-openai's ChatOpenAI
pointed at NVIDIA's base URL instead of OpenAI's or Z.ai's own endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import mimetypes
import re
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trip_planner.app")

# --------------------------------------------------------------------------
# Custom user prompt (plain txt file you edit yourself).
# Loaded from CUSTOM_PROMPT_FILE or ./my_prompt.txt and prepended to every
# AI prompt (chat, research, draft, intake). Empty file = built-in defaults.
# Re-read when the file mtime changes so you can edit without restarting.
# --------------------------------------------------------------------------
_CUSTOM_PROMPT_CACHE = {"mtime": 0.0, "text": "", "path": ""}


def _custom_prompt_path() -> str:
    return os.getenv(
        "CUSTOM_PROMPT_FILE", os.path.join(_THIS_DIR if "_THIS_DIR" in globals() else os.path.dirname(os.path.abspath(__file__)), "my_prompt.txt")
    )


def get_custom_prompt() -> str:
    path = _custom_prompt_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""
    if path != _CUSTOM_PROMPT_CACHE.get("path") or mtime != _CUSTOM_PROMPT_CACHE.get("mtime"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            _CUSTOM_PROMPT_CACHE.update({"mtime": mtime, "text": text, "path": path})
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
# Load mcp.py as "trip_mcp" (NOT as "mcp") -- see the long comment in
# mcp.py's _import_mcp_adapter() for why a plain `import mcp` here would be
# dangerous: it would shadow the real `mcp` SDK package for the rest of the
# process. This keeps our module out of sys.modules under that name.
# --------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("trip_mcp", os.path.join(_THIS_DIR, "mcp.py"))
trip_mcp = importlib.util.module_from_spec(_spec)
sys.modules["trip_mcp"] = trip_mcp  # register before exec so `from trip_mcp import ...` below resolves
_spec.loader.exec_module(trip_mcp)

from trip_mcp import (  # noqa: E402  (module built dynamically above)
    Activity,
    CityStop,
    DayPlan,
    Itinerary,
    TransferLeg,
    TravelMode,
    TripConstraints,
    ValidationReport,
    get_agent_tools,
    validate_itinerary,
)

# --------------------------------------------------------------------------
# LLM setup -- GLM-5.3 served via NVIDIA's free OpenAI-compatible API
# (build.nvidia.com / integrate.api.nvidia.com). One NVIDIA_API_KEY covers
# both the text model (planning/reasoning/tool-calling) and the multimodal
# Flash model (image analysis for uploads).
# --------------------------------------------------------------------------

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
GLM_PLANNING_MODEL = os.getenv(
    "GLM_PLANNING_MODEL", os.getenv("GLM_MODEL", "z-ai/glm-5.3")
)
GLM_VISION_MODEL = os.getenv("GLM_VISION_MODEL", "z-ai/glm-5.3-flash")
GLM_CHAT_MODEL = os.getenv("GLM_CHAT_MODEL", "z-ai/glm-5.3-flash")
GLM_CHAT_REASONING_EFFORT = os.getenv("GLM_CHAT_REASONING_EFFORT", "low")
GLM_CHAT_MAX_TOKENS = int(os.getenv("GLM_CHAT_MAX_TOKENS", "16384"))
GLM_CHAT_TIMEOUT_SECONDS = int(os.getenv("GLM_CHAT_TIMEOUT_SECONDS", "180"))
GLM_PLANNING_MAX_TOKENS = int(os.getenv("GLM_PLANNING_MAX_TOKENS", "16384"))

if not NVIDIA_API_KEY:
    logger.warning(
        "NVIDIA_API_KEY is not set -- LLM calls will fail until it is configured. "
        "Get a free key at https://build.nvidia.com/z-ai/glm-5-3"
    )


def _build_llm(model: str, temperature: float = 0.3, **client_options):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=NVIDIA_API_KEY or "unset",
        base_url=GLM_BASE_URL,
        temperature=temperature,
        **client_options,
    )


llm = _build_llm(GLM_PLANNING_MODEL, temperature=0.3, max_tokens=GLM_PLANNING_MAX_TOKENS)
chat_llm = _build_llm(
    GLM_CHAT_MODEL,
    temperature=0.4,
    streaming=True,
    max_tokens=GLM_CHAT_MAX_TOKENS,
    timeout=GLM_CHAT_TIMEOUT_SECONDS,
    max_retries=0,
    # NVIDIA's hosted endpoint rejects `clear_thinking`; send only the
    # supported reasoning-effort option so chat requests are accepted.
    extra_body={"reasoning_effort": GLM_CHAT_REASONING_EFFORT},
)

# Fast itinerary drafter for trip requests started in chat: the lighter model
# with low reasoning, so a full day-by-day plan comes back in seconds.
fast_llm = _build_llm(
    GLM_CHAT_MODEL,
    temperature=0.2,
    max_tokens=GLM_PLANNING_MAX_TOKENS,
    timeout=int(os.getenv("TRIP_FAST_TIMEOUT_SECONDS", "120")),
    max_retries=0,
    extra_body={"reasoning_effort": "low"},
)
TRIP_CHAT_FAST = os.getenv("TRIP_CHAT_FAST", "1") == "1"

# Live-research chat model: tool calling needs the full token budget
# (reasoning + a full day-by-day itinerary + citations).
GLM_AGENT_MODEL = os.getenv("GLM_AGENT_MODEL", GLM_PLANNING_MODEL)
GLM_AGENT_MAX_TOKENS = int(os.getenv("GLM_AGENT_MAX_TOKENS", "16384"))
GLM_AGENT_TIMEOUT_SECONDS = int(os.getenv("GLM_AGENT_TIMEOUT_SECONDS", "180"))
GLM_AGENT_REASONING_EFFORT = os.getenv("GLM_AGENT_REASONING_EFFORT", "low")

agent_llm = _build_llm(
    GLM_AGENT_MODEL,
    temperature=0.3,
    streaming=True,
    max_tokens=GLM_AGENT_MAX_TOKENS,
    timeout=GLM_AGENT_TIMEOUT_SECONDS,
    max_retries=1,
    extra_body={"reasoning_effort": GLM_AGENT_REASONING_EFFORT},
)

import live_chat  # noqa: E402  (tools + tool-calling loop for /chat)

# --------------------------------------------------------------------------
# Upload storage (images get a vision pass with GLM-5.3-Flash; files get
# best-effort text extraction). Swap this in-memory index + local disk for
# a real object store / DB in production.
# --------------------------------------------------------------------------

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
(UPLOAD_DIR / "images").mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "files").mkdir(parents=True, exist_ok=True)

attachments: Dict[str, dict] = {}  # attachment_id -> metadata


def _analyze_image_with_vision(path: Path) -> str:
    """Send the uploaded image to GLM-5.3-Flash (multimodal) and ask for a
    short, trip-relevant summary: what it shows and any details worth
    folding into the itinerary (a booking confirmation, a landmark photo
    used as inspiration, a visa page, etc.)."""
    try:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        vision_llm = _build_llm(GLM_VISION_MODEL, temperature=0.0)
        from langchain_core.messages import HumanMessage

        message = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "This image was uploaded as context for a multi-city trip "
                    "being planned. In 2-4 sentences, describe what it shows and "
                    "extract anything useful for trip planning: a place/landmark, "
                    "a booking confirmation's dates/cost/city, a menu, a visa or "
                    "passport page's relevant dates, etc. If it's just inspiration "
                    "(e.g. a photo of a destination), say what destination/vibe it "
                    "suggests."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])
        response = vision_llm.invoke([message])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.error("Vision analysis failed: %s", exc)
        return "(image uploaded; automatic analysis unavailable)"


def _extract_file_text(path: Path, content_type: str) -> str:
    """Best-effort text extraction for uploaded documents (e.g. an existing
    hotel/flight booking PDF). Falls back gracefully if pypdf is missing or
    the file isn't a format we know how to parse."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text[:5000]
        if suffix in (".txt", ".md", ".csv", ".json"):
            return path.read_text(errors="ignore")[:5000]
    except Exception as exc:
        logger.warning("Text extraction failed for %s: %s", path, exc)
    return "(no text extracted -- unsupported or unreadable format)"


# --------------------------------------------------------------------------
# Live status events (SSE)
# --------------------------------------------------------------------------
# One asyncio.Queue per in-flight job. plan_trip() and everything it calls
# push an event here *at the moment that real step actually starts* -- these
# are not simulated/timed phases, they're emitted from inside the real
# research-agent callbacks and the real validators. No HTTP route drains these
# queues any more (the job endpoints were removed); /trips/sync passes no job_id,
# so these emits are currently no-ops.
#
# Each status phase corresponds to one real backend action:
#   architecting    -- building the research agent (tools + prompt), once per process
#   orchestrating   -- plan_trip coordinating this iteration across cities/days
#   searching       -- the agent is calling the web_search tool
#   mapping         -- the agent is calling the estimate_route (geocoding) tool
#   wandering       -- the agent is calling any other tool (MCP time/fetch/public_apis)
#   perambulating   -- the agent's LLM is reasoning about what to do next
#   plotting        -- draft_phase: the structured LLM is building the day-by-day itinerary
#   calculating     -- validate_itinerary: checking dates/budget/daily schedule
#   navigating      -- validate_itinerary: checking city-to-city transfer feasibility
#   dilly dallying  -- validation failed; looping back for another research+draft pass
job_queues: Dict[str, "asyncio.Queue"] = {}


async def _emit(job_id: Optional[str], phase: str, detail: str = "") -> None:
    """Push a live status event from async code (plan_trip, research_phase, draft_phase)."""
    if not job_id:
        return
    queue = job_queues.get(job_id)
    if queue is None:
        return
    await queue.put({"phase": phase, "detail": detail, "ts": datetime.utcnow().isoformat()})


def _emit_sync(job_id: Optional[str], phase: str, detail: str = "") -> None:
    """Push a live status event from sync code (the validators, via on_phase)."""
    if not job_id:
        return
    queue = job_queues.get(job_id)
    if queue is None:
        return
    try:
        queue.put_nowait({"phase": phase, "detail": detail, "ts": datetime.utcnow().isoformat()})
    except asyncio.QueueFull:  # pragma: no cover -- unbounded queue, shouldn't happen
        logger.warning("Status queue full for job %s; dropping '%s' event", job_id, phase)


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------
# Research agent: web search + route estimator + any configured MCP tools
# --------------------------------------------------------------------------

_agent_executor = None


async def _get_research_agent():
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:  # langchain >= 1.0 moved the legacy agents
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools = await get_agent_tools()
    system_text = _with_custom_prompt(
        "You are a travel researcher. Given trip constraints (and, on a "
        "replanning pass, a list of problems with the previous plan), use "
        "your tools to gather real, current, specific facts: attraction "
        "names and opening hours, realistic ticket/meal/lodging costs in "
        "local currency, intercity transport options and durations, and any "
        "entry requirements. Prefer multiple targeted searches over one "
        "broad one. Use web_search with 10 results per query when available. "
        "Finish with a concise, well-organized briefing the "
        "planner can turn directly into a day-by-day itinerary -- do NOT "
        "write the itinerary yourself, just the research findings."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    _agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=int(os.getenv("RESEARCH_MAX_ITERATIONS", "5")),
    )
    return _agent_executor


_PhaseStatusCallbackCls = None


def _get_phase_callback_cls():
    """Lazily build the callback class (same lazy-import style as
    _get_research_agent above) -- a LangChain callback handler that turns
    the research agent's *real* tool calls and reasoning steps into live
    status events, fed straight from AgentExecutor's own callback hooks."""
    global _PhaseStatusCallbackCls
    if _PhaseStatusCallbackCls is not None:
        return _PhaseStatusCallbackCls

    from langchain_core.callbacks import AsyncCallbackHandler

    class PhaseStatusCallback(AsyncCallbackHandler):
        def __init__(self, job_id: Optional[str]):
            self.job_id = job_id

        async def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
            await _emit(self.job_id, "perambulating", "Agent is deciding its next research step")

        async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
            await _emit(self.job_id, "perambulating", "Agent is deciding its next research step")

        async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
            name = (serialized or {}).get("name", "")
            query = (input_str or "")[:140]
            if name == "web_search":
                await _emit(self.job_id, "searching", query)
            elif name == "estimate_route":
                await _emit(self.job_id, "mapping", query)
            else:
                await _emit(self.job_id, "wandering", f"{name}: {query}" if name else query)

    _PhaseStatusCallbackCls = PhaseStatusCallback
    return _PhaseStatusCallbackCls


async def research_phase(
    constraints: TripConstraints,
    attachment_context: str,
    feedback: Optional[List[str]] = None,
    job_id: Optional[str] = None,
) -> str:
    agent = await _get_research_agent()
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n\nThe previous itinerary draft had these problems -- research "
            "whatever is needed to fix them specifically:\n- " + "\n- ".join(feedback)
        )
    attachment_block = ("Attachment context:\n" + attachment_context) if attachment_context else ""
    task = (
        f"Trip constraints:\n{constraints.model_dump_json(indent=2)}\n"
        f"{attachment_block}"
        f"{feedback_block}"
    )
    config = None
    if job_id:
        callback_cls = _get_phase_callback_cls()
        config = {"callbacks": [callback_cls(job_id)]}
    # Bound the research loop so a hung LLM/tool can never stall planning
    # forever -- the job fails fast with a clear error instead.
    timeout_s = int(os.getenv("RESEARCH_TIMEOUT_SECONDS", "150"))
    try:
        result = await asyncio.wait_for(agent.ainvoke({"input": task}, config=config), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"Research took longer than {timeout_s}s; try again or set RESEARCH_MAX_ITERATIONS lower.") from exc
    return result.get("output", "")


async def draft_phase(
    constraints: TripConstraints,
    research_notes: str,
    attachment_context: str,
    feedback: Optional[List[str]] = None,
    job_id: Optional[str] = None,
    fast: bool = False,
) -> Itinerary:
    await _emit(job_id, "plotting", "Assembling the day-by-day itinerary")
    structured_llm = (fast_llm if fast else llm).with_structured_output(Itinerary)
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n\nThe previous draft failed validation for these reasons -- fix "
            "them in this new draft:\n- " + "\n- ".join(feedback)
        )
    prompt = _with_custom_prompt(
        "Build a complete, realistic day-by-day multi-city itinerary from the "
        "constraints and research below. Every day in the date range must "
        "appear exactly once. Respect the daily time window and pace. Include "
        "a `transfer` on any day the traveler changes cities, with realistic "
        "depart/arrive times for the chosen mode. Give every activity a "
        "realistic estimated_cost (0 is fine for free things) and, when the "
        "research specifies them, opening_hours as 'HH:MM-HH:MM'. Make "
        "total_estimated_cost equal to the sum of all lodging, transfer, and "
        "activity costs.\n\n"
        f"Constraints:\n{constraints.model_dump_json(indent=2)}\n\n"
        + ("No web research was run: rely on well-known, realistic general knowledge, "
           "keep estimates conservative, and omit opening_hours.\n\n" if fast else "")
        + f"Research findings:\n{research_notes or '(none)'}\n"
        f"{('Attachment context:' + attachment_context) if attachment_context else ''}"
        f"{feedback_block}"
    )
    return await structured_llm.ainvoke(prompt)


def _attachment_context(attachment_ids: List[str]) -> str:
    parts = []
    for aid in attachment_ids:
        meta = attachments.get(aid)
        if not meta:
            continue
        parts.append(f"- [{meta['kind']}] {meta['filename']}: {meta.get('analysis', '')}")
    return "\n".join(parts)


async def plan_trip(
    constraints: TripConstraints,
    max_iterations: int = 4,
    job_id: Optional[str] = None,
    fast: bool = False,
) -> dict:
    attachment_context = _attachment_context(constraints.attachment_ids)
    history = []
    feedback: Optional[List[str]] = None

    if not fast and _agent_executor is None:
        await _emit(job_id, "architecting", "Building the research agent and its tools")
    await _emit(
        job_id, "orchestrating",
        f"Coordinating a {constraints.trip_days}-day trip across {len(constraints.destinations)} cit"
        f"{'y' if len(constraints.destinations) == 1 else 'ies'}",
    )

    for i in range(1, max_iterations + 1):
        if i > 1:
            await _emit(job_id, "dilly dallying", f"Draft {i - 1} didn't pass validation -- taking another pass")
        notes = "" if fast else await research_phase(constraints, attachment_context, feedback, job_id=job_id)
        itinerary = await draft_phase(constraints, notes, attachment_context, feedback, job_id=job_id, fast=fast)
        report: ValidationReport = validate_itinerary(
            itinerary, constraints,
            on_phase=(lambda phase, detail="", _jid=job_id: _emit_sync(_jid, phase, detail)),
        )
        history.append({
            "iteration": i,
            "is_valid": report.is_valid,
            "issues": [issue.model_dump() for issue in report.issues],
            "total_cost": report.total_cost,
        })
        if report.is_valid:
            return {
                "status": "valid",
                "iterations": i,
                "itinerary": itinerary.model_dump(mode="json"),
                "validation": report.model_dump(),
                "history": history,
            }
        feedback = [issue.message for issue in report.issues if issue.severity == "error"]

    # Ran out of iterations -- return the best-effort last draft plus the
    # unresolved issues rather than silently failing.
    return {
        "status": "unresolved_after_max_iterations",
        "iterations": max_iterations,
        "itinerary": itinerary.model_dump(mode="json"),
        "validation": report.model_dump(),
        "history": history,
    }


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="Multi-City Trip Planning Agent", version="1.0.0")

# Permissive by default so the static frontend (any origin/file://) can open
# the SSE stream below. Tighten allow_origins to your real frontend's origin
# before deploying this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _warmup_agent_tools():
    """Pre-connect MCP servers at boot (in the background) so the first
    real request doesn't pay the `uvx` download/connect cost. Failures
    only log -- requests still work with web_search + route estimator."""
    async def _run():
        try:
            tools = await get_agent_tools()
            logger.info("Startup warmup: %d agent tool(s) ready.", len(tools))
        except Exception as exc:
            logger.warning("Startup warmup failed (non-fatal): %s", exc)

    asyncio.create_task(_run())


class TripRequest(TripConstraints):
    max_iterations: int = Field(default=4, ge=1, le=8)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    history: List[ChatTurn] = Field(default_factory=list)
    trip_pending: bool = False  # True while the planner is waiting for origin/destination


@app.get("/health")
async def health():
    custom_path = _custom_prompt_path()
    custom_loaded = bool(get_custom_prompt())
    return {
        "status": "ok",
        "model": GLM_PLANNING_MODEL,
        "planning_model": GLM_PLANNING_MODEL,
        "planning_max_tokens": GLM_PLANNING_MAX_TOKENS,
        "chat_model": GLM_CHAT_MODEL,
        "chat_reasoning_effort": GLM_CHAT_REASONING_EFFORT,
        "chat_max_tokens": GLM_CHAT_MAX_TOKENS,
        "chat_timeout_seconds": GLM_CHAT_TIMEOUT_SECONDS,
        "vision_model": GLM_VISION_MODEL,
        "live_chat_model": GLM_AGENT_MODEL,
        "live_search_backend": "tavily" if os.getenv("TAVILY_API_KEY") else "duckduckgo",
        "live_search_results": 10,
        "live_tools": [t.name for t in live_chat.TOOLS],
        "custom_prompt_file": custom_path,
        "custom_prompt_loaded": custom_loaded,
        "mcp_connect_timeout_seconds": int(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "30")),
    }


@app.get("/health/mcp")
async def health_mcp():
    """Real MCP diagnostics -- proves MCP servers actually connect."""
    from trip_mcp import get_mcp_status

    return await get_mcp_status()


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def frontend():
    """Serve the single-page frontend from the same Render web service."""
    index_file = Path(_THIS_DIR) / "frontend" / "index.html"
    return FileResponse(index_file, media_type="text/html")


class TripChatIntake(BaseModel):
    """Trip details extracted from a natural-language chat request."""
    origin_city: Optional[str] = None
    destinations: List[CityStop] = Field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_days: Optional[int] = Field(default=None, ge=1, le=30)
    total_budget: Optional[float] = Field(default=None, gt=0)
    currency: Optional[str] = None
    travelers: Optional[int] = Field(default=None, ge=1, le=20)
    pace: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


_TRIP_CANCEL_RE = re.compile(
    r"\b(?:cancel|stop|never mind|nevermind|forget it|skip it|don't(?: want to)? plan|"
    r"do not(?: want to)? plan|rather not plan)\b",
    re.I,
)


def _is_trip_cancellation(message: str) -> bool:
    return bool(_TRIP_CANCEL_RE.search(message or ""))


def _is_trip_planning_request(message: str) -> bool:
    """Detects an explicit request to plan a trip (same rules the UI used before)."""
    text = " ".join((message or "").split()).lower()
    if _is_trip_cancellation(text):
        return False
    action_before_trip = re.search(r"\b(?:plan|planning|create|build|make|organize|draft|design)\b.{0,60}\b(?:trip|travel|itinerary|vacation|holiday)\b", text)
    trip_before_action = re.search(r"\b(?:trip|travel|itinerary|vacation|holiday)\b.{0,60}\b(?:plan|planning|create|build|make|organize|draft)\b", text)
    itinerary_request = re.search(r"\b(?:\d+\s*[- ]day\s+)?itinerary\s+(?:for|to|in|around)\b", text)
    duration_plan = re.search(r"\b(?:plan|create|build|make|organize|draft)\b.{0,50}\b(?:\d+\s*(?:day|night)s?|weekend)\b.{0,50}\b(?:in|to|around)\b", text)
    return bool(action_before_trip or trip_before_action or itinerary_request or duration_plan)


def _data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _trip_request_from_chat(message: str, history: list) -> dict:
    """Extract constraints from chat. Origin and destination are never guessed;
    dates and budget use disclosed defaults."""
    from langchain_core.messages import HumanMessage, SystemMessage

    today = date.today()
    extraction_prompt = _with_custom_prompt(
        "Extract a trip request into the provided structured fields. Use only details "
        "stated in the current message or conversation history. Never invent an origin "
        "city or destination; leave either missing if unclear. Resolve relative dates "
        f"relative to today ({today.isoformat()}). Preserve the user's requested cities, "
        "country, duration, traveler count, currency, budget, pace and interests. Leave "
        "dates, duration and budget empty when absent; the server will apply and disclose "
        "sensible defaults.\n\n"
        f"Recent conversation: {json.dumps(history, ensure_ascii=False)}\n\n"
        f"Current trip request: {message}"
    )
    try:
        intake = await agent_llm.with_structured_output(TripChatIntake).ainvoke([
            SystemMessage(content=_with_custom_prompt("Extract trip details faithfully. Do not add unspecified locations.")),
            HumanMessage(content=extraction_prompt),
        ])
        if not isinstance(intake, TripChatIntake):
            intake = TripChatIntake.model_validate(intake)
    except Exception as exc:
        logger.exception("Could not extract trip details from chat")
        raise RuntimeError(
            "I couldn't read the trip details yet. Please try again or tell me your starting city and destination."
        ) from exc

    missing = []
    if not intake.origin_city or not intake.origin_city.strip():
        missing.append("starting city")
    if not intake.destinations:
        missing.append("destination")
    if missing:
        if len(missing) == 2:
            reply = "What city are you starting from, and where would you like to go?"
        elif missing[0] == "starting city":
            reply = "What city will you be starting from?"
        else:
            reply = "Where would you like to go?"
        return {"status": "clarification_required", "reply": reply}

    assumptions = []
    start_date = intake.start_date
    end_date = intake.end_date
    duration_days = intake.duration_days
    if start_date is None or end_date is None:
        if duration_days is None:
            duration_days = 3
            assumptions.append("3-day duration default")
        if start_date is None and end_date is None:
            start_date = today + timedelta(days=14)
            end_date = start_date + timedelta(days=duration_days - 1)
            assumptions.append(f"start date default: {start_date.isoformat()} (14 days from today)")
        elif start_date is None:
            start_date = end_date - timedelta(days=duration_days - 1)
        else:
            end_date = start_date + timedelta(days=duration_days - 1)
    if end_date < start_date:
        return {"status": "clarification_required", "reply": "Those dates appear to be in reverse order. What dates should I use?"}
    duration_days = (end_date - start_date).days + 1

    travelers = intake.travelers or 1
    currency = (intake.currency or "USD").upper()
    budget = intake.total_budget
    if intake.currency is None and intake.total_budget is not None:
        assumptions.append("currency default: USD")
    if budget is None:
        budget = max(1500.0, 500.0 * duration_days) * travelers
        assumptions.append(f"budget default: {currency} {budget:,.0f} total for {travelers} traveler(s)")
    if intake.travelers is None:
        assumptions.append("1 traveler (default)")
    pace = intake.pace or "moderate"
    if intake.pace is None:
        assumptions.append("moderate pace (default)")

    constraints = TripConstraints(
        origin_city=intake.origin_city.strip(),
        destinations=intake.destinations,
        start_date=start_date,
        end_date=end_date,
        total_budget=budget,
        currency=currency,
        travelers=travelers,
        pace=pace,
        preferences=intake.preferences,
        notes=intake.notes,
    )
    return {"status": "ready", "constraints": constraints, "assumptions": assumptions}


def _hhmm(value) -> str:
    return str(value or "")[:5]


def _format_trip_reply(plan: dict, assumptions: list) -> str:
    """Plain-text itinerary for the chat bubble (mirrors the old UI formatter)."""
    itinerary = plan.get("itinerary")
    if not itinerary:
        return "The trip planner finished, but did not return an itinerary."
    cur = itinerary.get("currency") or "USD"

    lines = [itinerary.get("trip_title") or "Your trip itinerary"]
    if assumptions:
        lines.append(f"Assumptions: {'; '.join(assumptions)}.")
    if itinerary.get("summary"):
        lines.append(itinerary["summary"])

    for index, day in enumerate(itinerary.get("days") or []):
        lines.append(f"\n{day.get('date') or f'Day {index + 1}'} — {day.get('city') or 'Destination'}")
        transfer = day.get("transfer")
        if transfer:
            lines.append(
                f"  Transfer: {transfer.get('from_city')} → {transfer.get('to_city')} by {transfer.get('mode')}, "
                f"{_hhmm(transfer.get('depart_time'))}–{_hhmm(transfer.get('arrive_time'))}."
            )
        for activity in day.get("activities") or []:
            time_range = f"{_hhmm(activity.get('start_time'))}–{_hhmm(activity.get('end_time'))}"
            cost = float(activity.get("estimated_cost") or 0)
            cost_txt = f" (est. {cur} {cost:.0f})" if cost > 0 else ""
            notes_txt = f" — {activity['notes']}" if activity.get("notes") else ""
            lines.append(f"  {time_range}: {activity.get('name')}{cost_txt}{notes_txt}")
        lodging = float(day.get("lodging_cost") or 0)
        if lodging > 0:
            lines.append(f"  Lodging estimate: {cur} {lodging:.0f}")

    total = float(itinerary.get("total_estimated_cost") or 0)
    lines.append(f"\nEstimated total: {cur} {total:.0f}.")
    issues = (plan.get("validation") or {}).get("issues") or []
    if issues:
        lines.append("Validation notes: " + " ".join(i.get("message", "") for i in issues[:4]))
    if plan.get("status") == "unresolved_after_max_iterations":
        lines.append("Some validation items may still need review.")
    return "\n".join(lines)


async def _stream_trip_chat(message: str, history: list):
    """SSE stream for a trip request made in chat: asks for missing places, or runs
    the real planner and streams its live phases, then the finished itinerary."""
    yield _data({"status": "Checking request", "detail": "Trip planning request"})
    yield _data({"status": "Reading trip details", "detail": "Extracting origin, destinations, dates and budget"})
    try:
        req = await _trip_request_from_chat(message, history)
    except Exception as exc:
        yield _data({"error": str(exc)})
        yield "data: [DONE]\n\n"
        return

    if req["status"] == "clarification_required":
        yield _data({"trip_pending": True})
        yield _data({"token": req["reply"]})
        yield "data: [DONE]\n\n"
        return

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    job_queues[job_id] = queue
    max_iterations = int(os.getenv("TRIP_MAX_ITERATIONS", "2"))
    task = asyncio.create_task(
        plan_trip(req["constraints"], max_iterations=max_iterations, job_id=job_id, fast=TRIP_CHAT_FAST)
    )
    try:
        yield _data({"trip_pending": False})
        while not task.done() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield _data({"status": item.get("phase", "working"), "detail": item.get("detail", "")})
        plan = task.result()
        issues = [i.get("message", "") for i in ((plan.get("validation") or {}).get("issues") or [])][:4]
        yield _data({
            "itinerary": plan.get("itinerary") or {},
            "assumptions": req["assumptions"],
            "validation_notes": issues,
            "plan_status": plan.get("status"),
        })
        yield _data({"token": _format_trip_reply(plan, req["assumptions"])})
    except Exception as exc:
        logger.exception("Trip planning from chat failed")
        yield _data({"error": f"Trip planning failed: {exc}"})
    finally:
        job_queues.pop(job_id, None)
        if not task.done():
            task.cancel()
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """Answer a free-form travel question with the configured chat model."""
    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI chat is not configured. Add NVIDIA_API_KEY to the Render service environment.",
        )

    history = [{"role": t.role, "content": t.content} for t in request.history[-10:]]
    headers = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # Trip planning requests (or a follow-up to an open trip question) run the
    # structured planner inside the chat stream. Cancelling falls through to chat.
    wants_trip = not _is_trip_cancellation(request.message) and (
        request.trip_pending or _is_trip_planning_request(request.message)
    )
    if wants_trip:
        return StreamingResponse(
            _stream_trip_chat(request.message, history), media_type="text/event-stream", headers=headers
        )

    # Travel / current-info questions -> live research (web search, weather,
    # exchange rates, real clock). Plain small talk -> the fast chat model.
    if live_chat.needs_live_research(request.message):

        async def stream_live():
            if request.trip_pending:
                yield _sse({"trip_pending": False})
            yield _sse({"status": "Checking request", "detail": "Travel or current-info question: using live tools"})
            try:
                async for event in live_chat.stream_live_chat(agent_llm, request.message, history):
                    yield _sse(event)
                yield "data: [DONE]\n\n"
            except Exception:
                logger.exception("Live chat stream failed")
                yield _sse({"error": "The AI request failed. Check NVIDIA_API_KEY and the model configuration in Render."})

        return StreamingResponse(stream_live(), media_type="text/event-stream", headers=headers)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=_with_custom_prompt(
        "You are TripCraft, a friendly assistant for natural, everyday conversation, "
        "with extra strength in travel. Reply directly to greetings and casual chat; "
        "do not turn simple chat into a travel-planning questionnaire. Keep normal "
        "replies short and easy to read, usually one to three sentences, without "
        "unnecessary preamble. When the user asks about travel, help with the "
        "requested planning and ask only the most useful follow-up question."
    ))]
    for turn in history:
        messages.append(HumanMessage(content=turn["content"]) if turn["role"] == "user"
                        else AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=request.message))

    async def stream_reply():
        if request.trip_pending:
            yield _sse({"trip_pending": False})
        yield _sse({"status": "Checking request", "detail": "General conversation: no live lookup needed"})
        yield _sse({"status": "Writing reply", "detail": f"model: {GLM_CHAT_MODEL}"})
        try:
            async for chunk in chat_llm.astream(messages):
                token = chunk.content
                if isinstance(token, list):
                    token = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in token
                    )
                elif not isinstance(token, str):
                    token = str(token or "")
                if token:
                    yield _sse({"token": token})
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("AI chat stream failed")
            yield _sse({"error": "The AI request failed. Check NVIDIA_API_KEY and the chat model configuration in Render."})

    return StreamingResponse(stream_reply(), media_type="text/event-stream", headers=headers)


# ---- uploads --------------------------------------------------------------


@app.post("/uploads/image")
async def upload_image(file: UploadFile = File(...)):
    attachment_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / "images" / f"{attachment_id}_{file.filename}"
    dest.write_bytes(await file.read())

    analysis = _analyze_image_with_vision(dest)
    attachments[attachment_id] = {
        "kind": "image",
        "filename": file.filename,
        "path": str(dest),
        "content_type": file.content_type,
        "analysis": analysis,
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    return {"attachment_id": attachment_id, "filename": file.filename, "analysis": analysis}


@app.post("/uploads/file")
async def upload_file(file: UploadFile = File(...)):
    attachment_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / "files" / f"{attachment_id}_{file.filename}"
    dest.write_bytes(await file.read())

    extracted_text = _extract_file_text(dest, file.content_type or "")
    attachments[attachment_id] = {
        "kind": "file",
        "filename": file.filename,
        "path": str(dest),
        "content_type": file.content_type,
        "analysis": extracted_text,
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    return {"attachment_id": attachment_id, "filename": file.filename, "extracted_text": extracted_text}


@app.get("/uploads/{attachment_id}")
async def get_upload(attachment_id: str):
    meta = attachments.get(attachment_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    return FileResponse(meta["path"], filename=meta["filename"])


@app.get("/uploads")
async def list_uploads():
    return {
        aid: {k: v for k, v in meta.items() if k != "path"}
        for aid, meta in attachments.items()
    }


# ---- trip planning ---------------------------------------------------------


@app.post("/trips/sync")
async def create_trip_sync(request: TripRequest):
    """Blocking variant: plans the trip and returns the final result directly.
    Handy for quick testing; prefer /trips for real deployments."""
    constraints = TripConstraints(**request.model_dump(exclude={"max_iterations"}))
    try:
        result = await plan_trip(constraints, max_iterations=request.max_iterations)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Trip planning failed")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
