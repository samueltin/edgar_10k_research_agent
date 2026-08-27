"""Component: Risk factors summarizer.

Splits Item 1A into its own category headings *before* summarizing, then
asks the LLM to summarize each category individually. This grounds every
summary bullet to a specific, verifiable slice of the source text, instead
of one LLM call over the whole section producing a summary a reviewer can
only spot-check by re-reading everything.
"""
import re
from typing import List
from pydantic import BaseModel, Field
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.llm_provider import get_llm

MAX_CONTEXT_CHARS = 100_000

# Item 1A's top-level category headings (e.g. "STRATEGIC AND COMPETITIVE
# RISKS") are consistently isolated as their own paragraph. The finer Title
# Case sub-headings underneath them are not -- some get merged into the
# following paragraph during HTML-to-text extraction -- so they aren't a
# reliable split boundary and are left as part of their category's body.
CATEGORY_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,&\-']{3,}$")
MAX_HEADING_CHARS = 100


class RiskCategorySummary(BaseModel):
    heading: str = Field(description="The risk category heading, copied exactly as provided")
    summary: str = Field(description="A 2-4 bullet point summary of this category's risks")


class RiskFactorsSummary(BaseModel):
    categories: List[RiskCategorySummary] = Field(
        description="One entry per category provided, in the same order"
    )


def split_by_category(risk_factors_text: str) -> List[dict]:
    """Split Item 1A into [{"heading": ..., "text": ...}, ...] chunks using
    the filing's own ALL-CAPS category headings as boundaries.
    """
    paragraphs = [p.strip() for p in risk_factors_text.split("\n\n") if p.strip()]

    chunks: List[dict] = []
    heading = "Overview"
    body: List[str] = []

    for p in paragraphs:
        if p.startswith("ITEM 1A"):
            continue
        if len(p) < MAX_HEADING_CHARS and CATEGORY_HEADING_RE.match(p):
            if body:
                chunks.append({"heading": heading, "text": "\n\n".join(body)})
            heading = p
            body = []
        else:
            body.append(p)

    if body:
        chunks.append({"heading": heading, "text": "\n\n".join(body)})

    return chunks


PROMPT_TEMPLATE = """You are an expert financial analyst. Below is {ticker}'s
Item 1A "Risk Factors" section, split into its category headings.

For EACH category listed, write a 2-4 bullet point summary of the risks it
describes. Return exactly one entry per category, in the same order given,
with the heading copied exactly as provided. Do not invent risks that are
not in the text, and do not merge categories together.

{chunks_text}
"""


def risk_summarizer_node(state: GraphState) -> dict:
    chunks = split_by_category(state["risk_factors_text"][:MAX_CONTEXT_CHARS])
    chunks_text = "\n\n".join(f"=== {c['heading']} ===\n{c['text']}" for c in chunks)

    llm = get_llm()
    structured_llm = llm.with_structured_output(RiskFactorsSummary)

    prompt = PROMPT_TEMPLATE.format(ticker=state["ticker"], chunks_text=chunks_text)
    result = structured_llm.invoke(prompt)

    risk_summary_by_category = [
        {"heading": chunk["heading"], "summary": category.summary, "source_text": chunk["text"]}
        for chunk, category in zip(chunks, result.categories)
    ]

    return {"risk_summary_by_category": risk_summary_by_category}
