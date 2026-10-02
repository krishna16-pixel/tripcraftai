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
    GLM_MODEL             default: z-ai/glm-5.3        (planning / reasoning / tool use)
    GLM_VISION_MODEL      default: z-ai/glm-5.3-flash  (multimodal -- used for image uploads)
    TAVILY_API_KEY        optional. Better web search than the DuckDuckGo fallback.
    MCP_SERVERS_JSON      optional. JSON config overriding the default free MCP
                           servers (see mcp.py -- time / fetch / public_apis, all
                           free, no key required).
    MCP_SERVERS_FILE      optional. Path to a JSON file with the same config.
    MCP_USE_DEFAULT_SERVERS  default: true. Set "false" to run with no MCP
                           servers instead of the free defaults.
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
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trip_planner.app")

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
GLM_MODEL = os.getenv("GLM_MODEL", "z-ai/glm-5.3")
GLM_VISION_MODEL = os.getenv("GLM_VISION_MODEL", "z-ai/glm-5.3-flash")

if not NVIDIA_API_KEY:
    logger.warning(
        "NVIDIA_API_KEY is not set -- LLM calls will fail until it is configured. "
        "Get a free key at https://build.nvidia.com/z-ai/glm-5-3"
    )


def _build_llm(model: str, temperature: float = 0.3):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=NVIDIA_API_KEY or "unset",
        base_url=GLM_BASE_URL,
        temperature=temperature,
    )


llm = _build_llm(GLM_MODEL, temperature=0.3)

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
# research-agent callbacks and the real validators. GET /trips/{job_id}/stream
# drains this queue and relays each event to the browser as SSE.
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

    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools = await get_agent_tools()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a travel researcher. Given trip constraints (and, on a "
            "replanning pass, a list of problems with the previous plan), use "
            "your tools to gather real, current, specific facts: attraction "
            "names and opening hours, realistic ticket/meal/lodging costs in "
            "local currency, intercity transport options and durations, and any "
            "entry requirements. Prefer multiple targeted searches over one "
            "broad one. Finish with a concise, well-organized briefing the "
            "planner can turn directly into a day-by-day itinerary -- do NOT "
            "write the itinerary yourself, just the research findings."
        )),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    _agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8)
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
    task = (
        f"Trip constraints:\n{constraints.model_dump_json(indent=2)}\n"
        f"{('Attachment context:\\n' + attachment_context) if attachment_context else ''}"
        f"{feedback_block}"
    )
    config = None
    if job_id:
        callback_cls = _get_phase_callback_cls()
        config = {"callbacks": [callback_cls(job_id)]}
    result = await agent.ainvoke({"input": task}, config=config)
    return result.get("output", "")


async def draft_phase(
    constraints: TripConstraints,
    research_notes: str,
    attachment_context: str,
    feedback: Optional[List[str]] = None,
    job_id: Optional[str] = None,
) -> Itinerary:
    await _emit(job_id, "plotting", "Assembling the day-by-day itinerary")
    structured_llm = llm.with_structured_output(Itinerary)
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n\nThe previous draft failed validation for these reasons -- fix "
            "them in this new draft:\n- " + "\n- ".join(feedback)
        )
    prompt = (
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
        f"Research findings:\n{research_notes}\n"
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


async def plan_trip(constraints: TripConstraints, max_iterations: int = 4, job_id: Optional[str] = None) -> dict:
    attachment_context = _attachment_context(constraints.attachment_ids)
    history = []
    feedback: Optional[List[str]] = None

    if _agent_executor is None:
        await _emit(job_id, "architecting", "Building the research agent and its tools")
    await _emit(
        job_id, "orchestrating",
        f"Coordinating a {constraints.trip_days}-day trip across {len(constraints.destinations)} cit"
        f"{'y' if len(constraints.destinations) == 1 else 'ies'}",
    )

    for i in range(1, max_iterations + 1):
        if i > 1:
            await _emit(job_id, "dilly dallying", f"Draft {i - 1} didn't pass validation -- taking another pass")
        notes = await research_phase(constraints, attachment_context, feedback, job_id=job_id)
        itinerary = await draft_phase(constraints, notes, attachment_context, feedback, job_id=job_id)
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

jobs: Dict[str, dict] = {}  # job_id -> {status, result, error}


class TripRequest(TripConstraints):
    max_iterations: int = Field(default=4, ge=1, le=8)


@app.get("/health")
async def health():
    return {"status": "ok", "model": GLM_MODEL, "vision_model": GLM_VISION_MODEL}


@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the single-page frontend from the same Render web service."""
    index_file = Path(_THIS_DIR) / "frontend" / "index.html"
    return FileResponse(index_file, media_type="text/html")


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


@app.post("/trips")
async def create_trip(request: TripRequest):
    """Kick off planning in the background; poll GET /trips/{job_id}, or
    watch it live via GET /trips/{job_id}/stream."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "result": None, "error": None}
    job_queues[job_id] = asyncio.Queue()

    constraints = TripConstraints(**request.model_dump(exclude={"max_iterations"}))

    async def _run():
        try:
            result = await plan_trip(constraints, max_iterations=request.max_iterations, job_id=job_id)
            jobs[job_id] = {"status": "done", "result": result, "error": None}
            await job_queues[job_id].put({"phase": "done", "detail": "", "ts": datetime.utcnow().isoformat()})
        except Exception as exc:  # pragma: no cover
            logger.exception("Trip planning failed")
            jobs[job_id] = {"status": "failed", "result": None, "error": str(exc)}
            await job_queues[job_id].put({"phase": "failed", "detail": str(exc), "ts": datetime.utcnow().isoformat()})
        finally:
            await job_queues[job_id].put(None)  # sentinel: tells the SSE stream to close

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/trips/{job_id}/stream")
async def stream_trip(job_id: str):
    """Real SSE endpoint: relays the live status events plan_trip() is
    actually emitting for this job (see the job_queues block above), then
    sends one final 'result' event with the job's outcome and closes."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        queue = job_queues.get(job_id)
        if queue is None:
            yield _sse_format("error", {"detail": "No live stream for this job."})
            return
        while True:
            item = await queue.get()
            if item is None:  # sentinel -- the background job is finished
                break
            event_name = "done" if item.get("phase") in ("done", "failed") else "status"
            yield _sse_format(event_name, item)
        job = jobs.get(job_id, {})
        yield _sse_format("result", job)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/trips/{job_id}")
async def get_trip(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


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
