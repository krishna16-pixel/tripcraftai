# =============================================================================
# TripCraft AI -- ONE speed for trip planning, with real live data
# (app.py patch -- supersedes the earlier app_py_timeout_fix.py; this file
# includes those two functions again, updated, plus four more)
#
# WHAT CHANGES:
#   - No more "fast skips research / slow does research" split. Trip
#     planning always runs ONE short, bounded live-research pass (web
#     prices, real FX rate via the new currency_exchange tool, weather/
#     time via your existing MCP servers), then drafts. Capped tightly so
#     it stays fast; if it still overruns, the same live-chat fallback as
#     before kicks in.
#   - The research agent now uses fast_llm (the light model) instead of
#     the full reasoning model, with max 3 tool calls and a 25s cap.
#   - draft_phase always uses fast_llm too -- no more switching models.
#   - If research times out, we don't fail the whole request -- we just
#     draft from general knowledge instead (old behaviour re-raised and
#     killed the whole plan; new behaviour degrades gracefully).
#
# REQUIRES: apply the mcp.py patch first (adds build_currency_exchange_tool
# and the updated get_agent_tools -- see mcp_py_currency_patch.py).
#
# HOW TO APPLY: replace these SIX functions in app.py with the versions
# below (same names/signatures unless noted). Everything else in app.py
# is untouched.
#   1. _get_research_agent
#   2. research_phase
#   3. draft_phase            (dropped the `fast` parameter)
#   4. plan_trip               (`fast` param kept but no longer branches
#                                anything -- safe no-op for compatibility)
#   5. _get_phase_callback_cls  (added a nicer status label for the new tool)
#   6. _trip_request_from_chat / _stream_trip_chat (unchanged from the
#      previous patch -- included again so this is the one file to apply)
# =============================================================================

import asyncio  # already imported in app.py -- listed here for clarity only

_INTAKE_TIMEOUT_SECONDS = 20      # extracting origin/dates/budget from chat
_PLANNER_DEADLINE_SECONDS = 55    # total time the whole plan (research+draft) gets
_PLANNER_MAX_ITERATIONS = 1       # one research+draft pass, not two


# -----------------------------------------------------------------------
# 1. Research agent -- now built on fast_llm, capped at 3 tool calls / 25s
# -----------------------------------------------------------------------
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
        "You are a travel researcher working under a tight time budget. Given "
        "trip constraints (and, on a replanning pass, a list of problems with "
        "the previous plan), make at most 3 targeted tool calls to gather the "
        "handful of facts that most change the plan: a realistic flight/"
        "transport cost range in local currency (web_search), the current "
        "exchange rate between the traveler's home currency and the "
        "destination currency (currency_exchange), and, if useful, expected "
        "weather or local time for the travel dates (the public_apis / time "
        "MCP tools, if available). Do not run more tool calls than necessary. "
        "Finish with a short, concrete briefing (bullet points, not prose) "
        "the planner can turn directly into a day-by-day itinerary -- do NOT "
        "write the itinerary yourself, just the research findings."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    # CHANGED: fast_llm instead of llm -- this pass always runs now, so it
    # has to stay quick.
    agent = create_tool_calling_agent(fast_llm, tools, prompt)
    _agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=int(os.getenv("RESEARCH_MAX_ITERATIONS", "3")),
        max_execution_time=float(os.getenv("RESEARCH_MAX_EXECUTION_SECONDS", "25")),
    )
    return _agent_executor


# -----------------------------------------------------------------------
# 2. research_phase -- lower timeout, degrade instead of failing the plan
# -----------------------------------------------------------------------
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

    # CHANGED: default lowered from 150s to 25s (also backstopped by the
    # agent's own max_execution_time above). CHANGED: on timeout we no
    # longer raise -- a slow research pass now just means the draft goes
    # out without live data, instead of killing the whole request.
    timeout_s = int(os.getenv("RESEARCH_TIMEOUT_SECONDS", "25"))
    try:
        result = await asyncio.wait_for(agent.ainvoke({"input": task}, config=config), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("Research pass exceeded %ds; drafting from general knowledge instead.", timeout_s)
        return ""
    except Exception:
        logger.exception("Research pass failed; drafting from general knowledge instead.")
        return ""
    return result.get("output", "")


# -----------------------------------------------------------------------
# 3. draft_phase -- always fast_llm, no more "fast" branch
# -----------------------------------------------------------------------
async def draft_phase(
    constraints: TripConstraints,
    research_notes: str,
    attachment_context: str,
    feedback: Optional[List[str]] = None,
    job_id: Optional[str] = None,
) -> Itinerary:
    await _emit(job_id, "plotting", "Assembling the day-by-day itinerary")
    structured_llm = fast_llm.with_structured_output(Itinerary)
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
        "activity costs. Use the research findings below -- especially any "
        "live flight-cost range, exchange rate, or weather -- instead of "
        "guessing wherever real data was found; if the research findings are "
        "empty, fall back to well-known, realistic general knowledge and keep "
        "estimates conservative.\n\n"
        f"Constraints:\n{constraints.model_dump_json(indent=2)}\n\n"
        f"Research findings:\n{research_notes or '(none)'}\n"
        f"{('Attachment context:' + attachment_context) if attachment_context else ''}"
        f"{feedback_block}"
    )
    return await structured_llm.ainvoke(prompt)


# -----------------------------------------------------------------------
# 4. plan_trip -- always does the (now bounded) research pass
# -----------------------------------------------------------------------
async def plan_trip(
    constraints: TripConstraints,
    max_iterations: int = 4,
    job_id: Optional[str] = None,
    fast: bool = True,  # kept for backward compatibility; no longer branches behaviour
) -> dict:
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

    return {
        "status": "unresolved_after_max_iterations",
        "iterations": max_iterations,
        "itinerary": itinerary.model_dump(mode="json"),
        "validation": report.model_dump(),
        "history": history,
    }


# -----------------------------------------------------------------------
# 5. Phase callback -- nicer status label for the new currency tool
# -----------------------------------------------------------------------
_PhaseStatusCallbackCls = None


def _get_phase_callback_cls():
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
            elif name == "currency_exchange":  # NEW
                await _emit(self.job_id, "converting", query)
            else:
                await _emit(self.job_id, "wandering", f"{name}: {query}" if name else query)

    _PhaseStatusCallbackCls = PhaseStatusCallback
    return _PhaseStatusCallbackCls


# -----------------------------------------------------------------------
# 6. Chat intake + SSE stream -- unchanged from the previous patch
# -----------------------------------------------------------------------
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
        intake = await asyncio.wait_for(
            fast_llm.with_structured_output(TripChatIntake).ainvoke([
                SystemMessage(content=_with_custom_prompt(
                    "Extract trip details faithfully. Do not add unspecified locations."
                )),
                HumanMessage(content=extraction_prompt),
            ]),
            timeout=_INTAKE_TIMEOUT_SECONDS,
        )
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


async def _stream_trip_chat(message: str, history: list):
    """SSE stream for a trip request made in chat: asks for missing places, or runs
    the real planner (now always with bounded live research) and streams its live
    phases, then the finished itinerary. If the planner still blows its deadline,
    falls through to the live-chat path so the user always gets a real answer."""
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
    constraints = req["constraints"]
    task = asyncio.create_task(
        plan_trip(constraints, max_iterations=_PLANNER_MAX_ITERATIONS, job_id=job_id)
    )

    deadline = asyncio.get_running_loop().time() + _PLANNER_DEADLINE_SECONDS
    plan = None
    try:
        yield _data({"trip_pending": False})
        while True:
            if task.done() and queue.empty():
                break
            if not task.done() and asyncio.get_running_loop().time() > deadline:
                task.cancel()
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield _data({"status": item.get("phase", "working"), "detail": item.get("detail", "")})

        if task.done() and not task.cancelled() and task.exception() is None:
            plan = task.result()
    except Exception:
        logger.exception("Trip planning task crashed")
    finally:
        job_queues.pop(job_id, None)
        if not task.done():
            task.cancel()

    if plan is not None:
        issues = [i.get("message", "") for i in ((plan.get("validation") or {}).get("issues") or [])][:4]
        yield _data({
            "itinerary": plan.get("itinerary") or {},
            "assumptions": req["assumptions"],
            "validation_notes": issues,
            "plan_status": plan.get("status"),
        })
        yield _data({"token": _format_trip_reply(plan, req["assumptions"])})
        yield "data: [DONE]\n\n"
        return

    # Still over deadline even with the bounded research pass -- fall back
    # to the live-tools chat path so the user still gets a real, streamed
    # answer instead of silence or a bare timeout error.
    yield _data({"status": "Switching to quick plan", "detail": "Structured planner was too slow"})
    fallback_msg = (
        f"{message}\n\nDates: {constraints.start_date} to {constraints.end_date}, "
        f"{constraints.travelers} traveler(s), budget {constraints.currency} {constraints.total_budget:,.0f}. "
        "Give a day-by-day plan with a budget breakdown, flight options, weather for those dates, "
        "and the current exchange rate."
    )
    try:
        async for event in live_chat.stream_live_chat(agent_llm, fallback_msg, history):
            yield _data(event)
    except Exception:
        logger.exception("Fallback chat failed")
        yield _data({"error": "The AI service is slow right now -- please try again in a minute."})
    yield "data: [DONE]\n\n"
