"""Component: Risk factors summarizer.

Splits Item 1A into its own category headings *before* summarizing, then
asks the LLM to summarize each category individually. This grounds every
summary bullet to a specific, verifiable slice of the source text, instead
of one LLM call over the whole section producing a summary a reviewer can
only spot-check by re-reading everything.
"""
from typing import List
from pydantic import BaseModel, Field
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.llm_provider import get_llm
from edgar_research_agent.agent.context_limits import get_max_input_chars

# BUG FIXED (found via a real Oracle 10-K run: the entire Item 1A section
# fell into a single "Overview" bucket instead of Oracle's real 6
# categories): the original CATEGORY_HEADING_RE only matched ALL-CAPS
# headings (e.g. MSFT's "STRATEGIC AND COMPETITIVE RISKS"). Oracle's real
# top-level categories are Title Case, not ALL CAPS (e.g. "Business and
# Operational Risks", "Legal and Regulatory Risks") -- none of them ever
# matched, so the heading variable in split_by_category() never changed
# from its "Overview" default, and every paragraph in the whole section
# landed in one chunk.
#
# Fix: is_category_heading() below accepts BOTH conventions, based on a
# distinguishing signal found across real MSFT and Oracle filings: a
# genuine top-level category heading has MOST of its content words
# capitalized (whether ALL CAPS like MSFT's, or genuine Title Case like
# Oracle's). This must NOT also match MSFT's finer, sentence-case
# sub-headings (e.g. "Competition in the technology sector", where only
# the first word is capitalized and real content words -- "technology",
# "sector" -- stay lowercase) -- those are deliberately left merged into
# their parent category's body, same as before this fix (see
# test_title_case_subheading_stays_inside_its_category_body).
MAX_HEADING_WORDS = 12
BULLET_PREFIXES = ("•", "◦", "‣", "·")
TERMINAL_PUNCTUATION = ".!?:;\""
MINOR_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "our", "us", "we", "with", "by", "as", "your", "its",
}
CONTENT_CAPITALIZATION_THRESHOLD = 0.7  # fraction of content words that must be capitalized


def is_category_heading(line: str) -> bool:
    """A top-level Item 1A category heading is short, has no terminal
    sentence punctuation, isn't a bullet or bare number, and has most of
    its CONTENT words (excluding short connector words like "and", "of",
    "the") capitalized -- true whether the filer uses ALL CAPS (MSFT) or
    genuine Title Case (Oracle). A sentence-case sub-heading, where only
    the first word is capitalized, fails this last check and is correctly
    left as part of its category's body.
    """
    p = line.strip()
    if not p or p.startswith(BULLET_PREFIXES):
        return False

    words = p.split()
    if len(words) < 2 or len(words) > MAX_HEADING_WORDS:
        return False

    if p[-1] in TERMINAL_PUNCTUATION:
        return False

    digits_only = p.replace(",", "").replace(".", "").strip()
    if digits_only.isdigit():
        return False

    return _mostly_capitalized_content_words(words)


def _mostly_capitalized_content_words(words: List[str]) -> bool:
    content_words = [w for w in words if w.lower().strip(",.;:'\"") not in MINOR_WORDS]
    if not content_words:
        return True  # e.g. a heading made entirely of connector words -- rare, don't reject
    capitalized = [w for w in content_words if w[0].isupper()]
    return (len(capitalized) / len(content_words)) >= CONTENT_CAPITALIZATION_THRESHOLD


def split_by_category(risk_factors_text: str) -> List[dict]:
    """Split Item 1A into [{"heading": ..., "text": ...}, ...] chunks using
    the filing's own category headings as boundaries (see
    is_category_heading() for what counts as a boundary, and why).
    """
    paragraphs = [p.strip() for p in risk_factors_text.split("\n\n") if p.strip()]

    chunks: List[dict] = []
    heading = "Overview"
    body: List[str] = []

    for p in paragraphs:
        if p.upper().startswith("ITEM 1A"):
            continue
        if is_category_heading(p):
            if body:
                chunks.append({"heading": heading, "text": "\n\n".join(body)})
            heading = p
            body = []
        else:
            body.append(p)

    if body:
        chunks.append({"heading": heading, "text": "\n\n".join(body)})

    return chunks


class RiskCategorySummary(BaseModel):
    heading: str = Field(description="The risk category heading, copied exactly as provided")
    summary: str = Field(description="A 2-4 bullet point summary of this category's risks")


class RiskFactorsSummary(BaseModel):
    categories: List[RiskCategorySummary] = Field(
        description="One entry per category provided, in the same order"
    )


PROMPT_TEMPLATE = """You are an expert financial analyst. Below is {ticker}'s
Item 1A "Risk Factors" section, split into its category headings.

For EACH category listed, write a 2-4 bullet point summary of the risks it
describes. Return exactly one entry per category, in the same order given,
with the heading copied exactly as provided. Do not invent risks that are
not in the text, and do not merge categories together.

{chunks_text}
"""

SKIPPED_SUMMARY_MESSAGE = "Summary skipped due to token usage management."


def risk_summarizer_node(state: GraphState) -> dict:
    """Summarize each risk category, with an optional cap on how many
    NAMED categories actually get sent to the LLM (max_categories_to_
    summarize in state -- None or absent means no limit, summarize
    everything).

    "Top N" means the first N NAMED categories in the FILING'S OWN
    document order (the order split_by_category() already returns them
    in), not ranked by any notion of importance -- there's no existing
    signal in this pipeline for ranking categories by materiality, and
    document order is the only ordering that's genuinely there.

    "Overview" (split_by_category()'s bucket for intro text before the
    first real heading) is DELIBERATELY EXEMPT from the limit and is
    always summarized when present. It isn't one of the company's actual
    risk categories, just leftover boilerplate -- counting it against
    the limit would mean "top 3" silently summarizes only 2 real
    categories, which isn't what an analyst asking for the top 3 risk
    factors actually means.

    0 means summarize none of the named categories (Overview is still
    exempt, per above). 999 is the UI's "summarize all" value -- real
    filings have far fewer named categories than that, so the plain
    named_chunks[:max_categories] slice below already summarizes
    everything without needing a dedicated "unlimited" sentinel.

    Categories beyond the limit are NOT sent to the LLM at all (the real
    point of this control -- saving the actual token cost, not just
    hiding the result) and get a placeholder summary instead, marked
    skipped=True. Their real source_text is still included, so the
    analyst can still read the original filing content for a skipped
    category themselves, even without an LLM summary of it. Downstream,
    groundedness_checker_node also skips these -- checking a placeholder
    message against real source text would be meaningless and would
    waste exactly the LLM calls this control exists to save.

    REAL BUG FOUND running a local Ollama model on real hardware (an 8GB
    GPU, 4096-token confirmed context window): sending the entire
    combined chunks_text (previously capped at a flat 100,000 characters,
    sized for cloud models) silently truncated far more than the local
    model's context window could hold. get_max_input_chars() now applies
    a provider- and task-aware limit instead -- see context_limits.py.
    max_categories_to_summarize is a SECOND, independent way to control
    how much text reaches the LLM in this same call -- both work together,
    the char limit truncates raw text length, the category limit reduces
    how many categories' worth of text are included in the first place.
    """
    max_input_chars = get_max_input_chars("risk_summarization")
    chunks = split_by_category(state["risk_factors_text"][:max_input_chars])

    overview_chunks = [c for c in chunks if c["heading"] == "Overview"]
    named_chunks = [c for c in chunks if c["heading"] != "Overview"]

    max_categories = state.get("max_categories_to_summarize")
    if max_categories is not None:
        named_chunks_to_summarize = named_chunks[:max_categories]
        skipped_chunks = named_chunks[max_categories:]
    else:
        named_chunks_to_summarize = named_chunks
        skipped_chunks = []

    chunks_to_summarize = overview_chunks + named_chunks_to_summarize

    risk_summary_by_category: List[dict] = []

    if chunks_to_summarize:
        chunks_text = "\n\n".join(f"=== {c['heading']} ===\n{c['text']}" for c in chunks_to_summarize)

        llm = get_llm()
        structured_llm = llm.with_structured_output(RiskFactorsSummary)

        prompt = PROMPT_TEMPLATE.format(ticker=state["ticker"], chunks_text=chunks_text)
        result = structured_llm.invoke(prompt)

        risk_summary_by_category.extend(
            {"heading": chunk["heading"], "summary": category.summary, "source_text": chunk["text"], "skipped": False}
            for chunk, category in zip(chunks_to_summarize, result.categories)
        )

    risk_summary_by_category.extend(
        {"heading": chunk["heading"], "summary": SKIPPED_SUMMARY_MESSAGE, "source_text": chunk["text"], "skipped": True}
        for chunk in skipped_chunks
    )

    return {"risk_summary_by_category": risk_summary_by_category}
