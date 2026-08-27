"""Component: Human review flag.

Marks a filing's segment data as flagged for manual review when the
validator does not pass. In a full UI this would surface in a review queue
rather than just printing.
"""
from edgar_research_agent.agent.state import GraphState


def human_review_node(state: GraphState) -> GraphState:
    print(f"FLAGGED FOR REVIEW ({state['ticker']}): {state['errors']}")
    return state
