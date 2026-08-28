"""Turns a research agent run's raw output into an analyst-ready memo.

This is where actual formatting/derivation happens: computed fields (like
gross margin %) and the memo layout the UI renders. Running the agent
itself is agent.graph.run()'s job, not this module's -- this module only
shapes what run() returns.
"""


def generate_memo(ticker: str, max_categories_to_summarize: int | None = None) -> dict:
    """Run the agent for one ticker and format the result into a memo.

    The agent.graph import is local to this function (not module-level), so
    the pure formatting helpers below stay testable without langgraph
    installed.

    max_categories_to_summarize is a cost control: only the first N risk
    categories (in the filing's own document order) get sent to the LLM
    for summarization; the rest are marked skipped, with a placeholder
    summary but their real source_text still included, so the analyst can
    still read the original content. None (default) means no limit. See
    agent/nodes/risk_summarizer.py for the full reasoning.

    Structure:
        {
            "company_name": str,
            "kpi_records": [KPIRecord, ...],
            "gross_margin_pct": dict[int, float] | None,  # fiscal_year -> %
            "segments": [SegmentKPI, ...],
            "validation_status": "PASS" | "FAIL",
            "errors": str,
            "risk_summary_by_category": [
                {"heading": str, "summary": str, "source_text": str, "skipped": bool, "groundedness": GroundednessResult},
                ...
            ],
            "mda_text": str,
        }

    groundedness is now computed automatically by the graph (agent.nodes.
    groundedness_checker), one per category, merged in here by position --
    see that node's docstring for the real cost/latency consequence of
    this running on every memo generation, not on demand. Skipped
    categories get a skipped=True GroundednessResult, not a real check.
    """
    from edgar_research_agent.agent.graph import run

    result = run(ticker, max_categories_to_summarize=max_categories_to_summarize)

    risk_summary_by_category = [
        {**category, "groundedness": groundedness}
        for category, groundedness in zip(result["risk_summary_by_category"], result["groundedness_results"])
    ]

    return {
        "company_name": result["company_name"],
        "kpi_records": result["kpi_records"],
        "gross_margin_pct": _compute_gross_margin_pct(result["kpi_records"]),
        "segments": result["extracted_segments"],
        "validation_status": result["validation_status"],
        "errors": result["errors"],
        "risk_summary_by_category": risk_summary_by_category,
        "mda_text": result["mda_text"],
    }


def _compute_gross_margin_pct(kpi_records) -> dict:
    """Derive gross margin % per fiscal year from Revenue and GrossProfit.

    Neither figure alone is "margin" -- this is genuine formatting logic,
    not a pass-through of what the graph already produced.
    """
    revenue_by_year = {r.fiscal_year: r.value for r in kpi_records if r.metric_name == "Revenue"}
    gross_profit_by_year = {r.fiscal_year: r.value for r in kpi_records if r.metric_name == "GrossProfit"}

    return {
        year: round(gross_profit_by_year[year] / revenue_by_year[year] * 100, 1)
        for year in revenue_by_year
        if year in gross_profit_by_year and revenue_by_year[year]
    }


def check_category_groundedness(category: dict):
    """Re-check one risk-summary category's groundedness on demand.

    generate_memo() now computes a groundedness result for every category
    automatically (see agent/nodes/groundedness_checker.py), so this is no
    longer the only way to get one. Its real purpose now: LLM output isn't
    perfectly deterministic, so an analyst who wants a second opinion on
    one specific category (rather than trusting the single automatic
    result) can call this to re-run the check. Kept as a thin wrapper
    around agent.groundedness.check_groundedness() for the same testability
    and reuse reasons as before.

    Skipped categories (see risk_summarizer.py's max_categories_to_
    summarize control) have no real summary to check -- re-checking one
    would waste an LLM call comparing a placeholder message against real
    source text, exactly the cost this control exists to avoid. Returns a
    skipped result immediately instead, without calling check_groundedness
    at all.

    category is one entry from risk_summary_by_category, i.e.
    {"heading": str, "summary": str, "source_text": str, "skipped": bool}.
    """
    from edgar_research_agent.agent.groundedness import check_groundedness, GroundednessResult

    if category.get("skipped"):
        return GroundednessResult(grounded=True, backend="skipped_no_summary", skipped=True)

    return check_groundedness(category["summary"], category["source_text"])
