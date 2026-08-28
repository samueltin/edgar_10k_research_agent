"""Builds and compiles the research agent's LangGraph workflow.

All five components from the Component diagram are graph nodes, including
XBRL extraction. xbrl_total_revenue is produced by extract_xbrl and flows
through state to the validator like everything else -- nothing is seeded
into state before graph.invoke() is called.

check_groundedness (agent/groundedness.py) now runs automatically here,
via check_groundedness -- see groundedness_checker.py's module docstring
for the real cost/latency consequence of this being wired into the
pipeline rather than left as an on-demand UI action.
"""
from langgraph.graph import StateGraph, END
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.nodes.xbrl_extractor import xbrl_extractor_node
from edgar_research_agent.agent.nodes.section_extractor import section_extractor_node
from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node
from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node
from edgar_research_agent.agent.nodes.groundedness_checker import groundedness_checker_node
from edgar_research_agent.agent.nodes.validator import validator_node, route_validation
from edgar_research_agent.agent.nodes.human_review import human_review_node


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("extract_xbrl", xbrl_extractor_node)
    workflow.add_node("fetch_sections", section_extractor_node)
    workflow.add_node("extract_segments", segment_extractor_node)
    workflow.add_node("summarize_risks", risk_summarizer_node)
    workflow.add_node("check_groundedness", groundedness_checker_node)
    workflow.add_node("validate", validator_node)
    workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("extract_xbrl")
    workflow.add_edge("extract_xbrl", "fetch_sections")
    workflow.add_edge("fetch_sections", "extract_segments")
    workflow.add_edge("fetch_sections", "summarize_risks")
    workflow.add_edge("summarize_risks", "check_groundedness")
    workflow.add_edge("extract_segments", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {"end": END, "human_review": "human_review"},
    )

    return workflow.compile()


def run(ticker: str, max_categories_to_summarize: int | None = None) -> dict:
    """Orchestration entry point: build the graph and run it for one ticker.

    Returns the raw GraphState result. This is the only place that knows
    about build_graph()/invoke() -- callers (e.g. memo/formatter.py) should
    use this rather than compiling the graph themselves.

    max_categories_to_summarize is a cost control -- see risk_summarizer.py.
    None (the default) means no limit, summarize every category found.
    0 means summarize none (every named category is skipped; Overview is
    still exempt, see risk_summarizer.py). 999 is the UI's "summarize all"
    value -- real filings have far fewer named risk categories than that,
    so it behaves the same as no limit without needing special-casing here.
    """
    graph = build_graph()
    initial_state = {"ticker": ticker}
    if max_categories_to_summarize is not None:
        initial_state["max_categories_to_summarize"] = max_categories_to_summarize
    return graph.invoke(initial_state)
