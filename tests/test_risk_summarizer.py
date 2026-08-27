"""Tests for the risk factors category-splitting logic -- the deterministic
part of risk_summarizer.py that grounds each LLM summary to a specific
slice of the source text."""
from edgar_research_agent.agent.nodes.risk_summarizer import split_by_category

SAMPLE_TEXT = """ITEM 1A. RISK FACTORS

Our operations and financial results are subject to various risks.

STRATEGIC AND COMPETITIVE RISKS

We face intense competition across all markets for our products.

Competition in the technology sector

Our competitors range in size from diversified global companies.

OPERATIONAL RISKS

Our business relies on our ability to attract and retain talent."""


def test_splits_on_all_caps_category_headings():
    chunks = split_by_category(SAMPLE_TEXT)

    headings = [c["heading"] for c in chunks]
    assert headings == ["Overview", "STRATEGIC AND COMPETITIVE RISKS", "OPERATIONAL RISKS"]


def test_title_case_subheading_stays_inside_its_category_body():
    chunks = split_by_category(SAMPLE_TEXT)

    strategic = next(c for c in chunks if c["heading"] == "STRATEGIC AND COMPETITIVE RISKS")
    assert "Competition in the technology sector" in strategic["text"]
    assert "diversified global companies" in strategic["text"]


def test_item_heading_line_is_dropped_not_treated_as_a_category():
    chunks = split_by_category(SAMPLE_TEXT)

    assert all("ITEM 1A" not in c["heading"] for c in chunks)


def test_overview_chunk_holds_text_before_first_category():
    chunks = split_by_category(SAMPLE_TEXT)

    overview = chunks[0]
    assert overview["heading"] == "Overview"
    assert "various risks" in overview["text"]
