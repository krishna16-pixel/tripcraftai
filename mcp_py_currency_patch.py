# =============================================================================
# TripCraft AI -- live currency-exchange tool (mcp.py patch)
#
# WHY: you already have free MCP tools for time (mcp-server-time) and
# weather/country data (public_apis -> Open-Meteo + REST Countries), but
# nothing gives the agent a real, current exchange rate -- it was guessing.
# There is no free, keyless flight-price API, so flight cost stays covered
# by the existing web_search tool (real search results, same as a human
# googling fares) -- a "free flights MCP" doesn't honestly exist.
#
# WHAT THIS ADDS: a dependency-free (stdlib-only, no new pip installs)
# currency_exchange tool using two free, keyless providers:
#   1. https://api.frankfurter.app  (ECB daily reference rates)
#   2. https://open.er-api.com      (fallback if Frankfurter is down)
#
# HOW TO APPLY:
#   1. Paste `build_currency_exchange_tool()` below into mcp.py, anywhere
#      near `build_route_estimator_tool()` (section 4, "Dependency-free
#      geo / route estimation" -- same style/placement).
#   2. Replace the existing `get_agent_tools()` function (section 3) with
#      the version below -- it just adds one tool to the list.
# =============================================================================


def build_currency_exchange_tool():
    """Free, keyless live currency conversion. Primary: Frankfurter (ECB
    reference rates, no key, no rate limit for normal use). Fallback:
    open.er-api.com. stdlib-only (urllib) so no new dependency is needed."""
    from langchain_core.tools import Tool

    def _run(query: str) -> str:
        """Input format: 'amount FROM to TO', e.g. '1 USD to INR' or
        'EUR to JPY' (amount defaults to 1 if omitted)."""
        import json as _json
        import re
        import urllib.request

        m = re.match(
            r"\s*([\d.,]+)?\s*([A-Za-z]{3})\s*(?:to|->|in)\s*([A-Za-z]{3})\s*$",
            query.strip(),
        )
        if not m:
            return (
                "Invalid input. Use: 'amount FROM to TO', e.g. '1 USD to INR' "
                "or 'EUR to JPY'."
            )
        amount_str, frm, to = m.groups()
        amount = float(amount_str.replace(",", "")) if amount_str else 1.0
        frm, to = frm.upper(), to.upper()

        def _fetch(url: str) -> dict:
            with urllib.request.urlopen(url, timeout=8) as resp:
                return _json.loads(resp.read().decode("utf-8"))

        try:
            data = _fetch(
                f"https://api.frankfurter.app/latest?amount={amount}&from={frm}&to={to}"
            )
            rate_value = (data.get("rates") or {}).get(to)
            if rate_value is None:
                raise ValueError(f"No rate returned for {frm}->{to}")
            unit_rate = rate_value / amount
            return (
                f"{amount:g} {frm} = {rate_value:,.2f} {to} "
                f"(1 {frm} = {unit_rate:,.4f} {to}) as of {data.get('date')}. "
                "Source: Frankfurter / ECB reference rates."
            )
        except Exception as exc1:
            try:
                data = _fetch(f"https://open.er-api.com/v6/latest/{frm}")
                unit_rate = (data.get("rates") or {}).get(to)
                if unit_rate is None:
                    raise ValueError("fallback provider returned no rate")
                total = unit_rate * amount
                return (
                    f"{amount:g} {frm} = {total:,.2f} {to} (1 {frm} = {unit_rate:,.4f} {to}). "
                    "Source: open.er-api.com (fallback)."
                )
            except Exception as exc2:
                logger.warning("Currency exchange tool failed: %s / fallback: %s", exc1, exc2)
                return (
                    f"Live exchange rate lookup failed ({exc1}). Use web_search for a "
                    "current rate instead, or clearly flag any estimate as unverified."
                )

    return Tool(
        name="currency_exchange",
        description=(
            "Get a real, current currency conversion using 3-letter ISO codes. "
            "Input: 'amount FROM to TO', e.g. '1 USD to INR' or '500 EUR to JPY' "
            "(amount defaults to 1 if omitted). Always use this instead of "
            "guessing an exchange rate when converting the traveler's budget."
        ),
        func=_run,
    )


# REPLACES the existing get_agent_tools() -- same function, one line added.
_cached_tools = None  # (already declared above get_agent_tools() in mcp.py -- do not duplicate)


async def get_agent_tools() -> list:
    """All tools exposed to the research agent: web search + route
    estimator + live currency exchange + whatever MCP servers are
    configured (time, fetch, public_apis by default). Cached after first
    successful load so we don't reconnect to MCP servers on every request."""
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools
    tools = [
        build_web_search_tool(),
        build_route_estimator_tool(),
        build_currency_exchange_tool(),  # NEW
    ]
    tools.extend(await load_mcp_tools())
    _cached_tools = tools
    return tools
