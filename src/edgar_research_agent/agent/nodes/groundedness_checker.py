"""Component: Groundedness checker.

Runs after risk_summarizer_node: checks each risk-category summary
against its own source text using agent.groundedness.check_groundedness(),
and stores one GroundednessResult per category in state, in the same
order as risk_summary_by_category.

This runs AUTOMATICALLY for every category on every memo generation,
by design (wired into the graph, not behind a UI button) -- which means
one extra LLM call per risk category, every time. For a company with 6-7
categories (e.g. Oracle), that's 6-7 additional calls added to every
single run, whether the analyst ever looks at the results or not. This
is a deliberate cost/latency trade-off, not a free improvement -- worth
remembering if the number of categories or run frequency grows.

Categories marked skipped=True by risk_summarizer_node's
max_categories_to_summarize control are skipped here too, not checked --
there's no real summary to verify (just a placeholder message), so
running a groundedness check against it would be meaningless AND would
waste exactly the LLM calls the summarization limit exists to save.
"""
from typing import List
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.groundedness import check_groundedness, GroundednessResult


def groundedness_checker_node(state: GraphState) -> dict:
    groundedness_results: List[GroundednessResult] = []

    for category in state["risk_summary_by_category"]:
        if category.get("skipped"):
            groundedness_results.append(
                GroundednessResult(grounded=True, backend="skipped_no_summary", skipped=True)
            )
        else:
            groundedness_results.append(
                check_groundedness(category["summary"], category["source_text"])
            )

    return {"groundedness_results": groundedness_results}
