"""Component: Validator.

Compares the LLM-extracted segment revenue sum against the XBRL consolidated
total. Uses a relative threshold rather than a fixed dollar amount, since a
fixed threshold is meaningless across companies of very different sizes.
"""
from edgar_research_agent.agent.state import GraphState

RELATIVE_VARIANCE_THRESHOLD = 0.01  # 1% of total revenue


def validator_node(state: GraphState) -> dict:
    total_extracted = sum(seg.revenue for seg in state["extracted_segments"])
    xbrl_total = state["xbrl_total_revenue"]

    if xbrl_total == 0:
        return {"validation_status": "FAIL", "errors": "No XBRL total revenue to validate against."}

    variance_pct = abs(total_extracted - xbrl_total) / xbrl_total

    if variance_pct <= RELATIVE_VARIANCE_THRESHOLD:
        return {"validation_status": "PASS", "errors": ""}

    error_msg = (
        f"Validation FAIL: XBRL total {xbrl_total:,.0f} vs LLM segment sum "
        f"{total_extracted:,.0f} ({variance_pct:.1%} variance)"
    )
    return {"validation_status": "FAIL", "errors": error_msg}


def route_validation(state: GraphState) -> str:
    return "end" if state["validation_status"] == "PASS" else "human_review"
