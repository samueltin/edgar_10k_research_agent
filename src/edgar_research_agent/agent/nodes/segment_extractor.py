"""Component: Segment extractor.

Uses an LLM with structured output to extract segment-level revenue from
MD&A / footnote text. Deliberately does NOT instruct the model to force its
numbers to reconcile with the total -- that would undermine the validator's
job of catching real discrepancies honestly.
"""
from edgar_research_agent.agent.schemas import SegmentExtraction
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.llm_provider import get_llm

MAX_CONTEXT_CHARS = 100_000

PROMPT_TEMPLATE = """You are an expert financial analyst. Extract the
segment-level revenue for {ticker} from the following MD&A / footnote text.
Report the segments exactly as described in the text, with their reported
revenue figures. Do not adjust figures to make them sum to any particular
total -- report what the text actually states.

Financial tables are commonly presented "in millions" or "in thousands" --
check the table header or surrounding text for this scale indicator, and
report each segment's revenue as a full USD amount (e.g. a table stated "in
millions" showing 139,996 must be reported as 139996000000).

Text:
{mda_text}
"""


def segment_extractor_node(state: GraphState) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(SegmentExtraction)

    prompt = PROMPT_TEMPLATE.format(
        ticker=state["ticker"],
        mda_text=state["mda_text"][:MAX_CONTEXT_CHARS],
    )

    result = structured_llm.invoke(prompt)
    return {"extracted_segments": result.segments, "validation_status": "PENDING"}
