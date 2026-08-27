"""Turns a research agent run's raw output into an analyst-ready memo.

This is where actual formatting/derivation happens: computed fields (like
gross margin %) and the memo layout the UI renders. Running the agent
itself is agent.graph.run()'s job, not this module's -- this module only
shapes what run() returns.
"""


def generate_memo(ticker: str) -> dict:
    """Run the agent for one ticker and format the result into a memo.

    The agent.graph import is local to this function (not module-level), so
    the pure formatting helpers below stay testable without langgraph
    installed.

    Structure:
        {
            "company_name": str,
            "kpi_records": [KPIRecord, ...],
            "gross_margin_pct": dict[int, float] | None,  # fiscal_year -> %
            "segments": [SegmentKPI, ...],
            "validation_status": "PASS" | "FAIL",
            "errors": str,
            "risk_summary_by_category": [{"heading": str, "summary": str, "source_text": str}, ...],
            "mda_text": str,
        }
    """
    from edgar_research_agent.agent.graph import run

    result = run(ticker)

    return {
        "company_name": result["company_name"],
        "kpi_records": result["kpi_records"],
        "gross_margin_pct": _compute_gross_margin_pct(result["kpi_records"]),
        "segments": result["extracted_segments"],
        "validation_status": result["validation_status"],
        "errors": result["errors"],
        "risk_summary_by_category": result["risk_summary_by_category"],
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
