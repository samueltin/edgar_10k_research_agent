"""Component: Section extractor.

Pulls MD&A (Item 7) and Risk Factors (Item 1A) text from the latest 10-K.
"""
from edgar_research_agent.edgar_client.filing_sections import get_mda_and_risk_factors
from edgar_research_agent.agent.state import GraphState


def section_extractor_node(state: GraphState) -> dict:
    mda_text, risk_factors_text = get_mda_and_risk_factors(state["ticker"])
    return {"mda_text": mda_text, "risk_factors_text": risk_factors_text}
