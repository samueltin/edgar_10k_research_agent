"""Tests for the risk factors category-splitting logic -- the deterministic
part of risk_summarizer.py that grounds each LLM summary to a specific
slice of the source text."""
from edgar_research_agent.agent.nodes.risk_summarizer import split_by_category, is_category_heading

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


# --- Regression tests: real bug found via a real Oracle 10-K run ---
# Before the fix, split_by_category() returned exactly ONE chunk
# ("Overview") for Oracle's entire Item 1A, since none of its real
# Title Case headings matched the old ALL-CAPS-only regex.

ORACLE_SAMPLE_TEXT = """ITEM 1A. RISK FACTORS

We operate in rapidly changing economic and technological environments.

Business and Operational Risks

We may be unsuccessful in developing and selling new products and services.

Our AI products may not operate as anticipated, which could adversely affect our reputation.

Data Privacy, Cybersecurity and Intellectual Property Risks

We are subject to business, financial and reputational risks related to cybersecurity incidents.

Legal and Regulatory Risks

Adverse litigation results could affect our business.

Financial Risks

Our operations can be difficult for us to predict because our quarterly results may fluctuate.

Risks Related to our Common and Preferred Stock

Our stock price could become more volatile and your investment could lose value.

General Risks

Economic, political and market conditions can adversely affect our business."""


def test_real_oracle_headings_are_each_detected_as_their_own_category():
    chunks = split_by_category(ORACLE_SAMPLE_TEXT)

    headings = [c["heading"] for c in chunks]
    assert headings == [
        "Overview",
        "Business and Operational Risks",
        "Data Privacy, Cybersecurity and Intellectual Property Risks",
        "Legal and Regulatory Risks",
        "Financial Risks",
        "Risks Related to our Common and Preferred Stock",
        "General Risks",
    ]


def test_real_oracle_categories_hold_their_own_body_text_only():
    chunks = split_by_category(ORACLE_SAMPLE_TEXT)

    financial = next(c for c in chunks if c["heading"] == "Financial Risks")
    assert "quarterly results" in financial["text"]
    assert "cybersecurity incidents" not in financial["text"]
    assert "litigation" not in financial["text"]


def test_is_category_heading_accepts_real_oracle_title_case_headings():
    real_oracle_headings = [
        "Business and Operational Risks",
        "Data Privacy, Cybersecurity and Intellectual Property Risks",
        "Legal and Regulatory Risks",
        "Financial Risks",
        "Risks Related to our Common and Preferred Stock",
        "General Risks",
    ]
    for heading in real_oracle_headings:
        assert is_category_heading(heading), f"{heading!r} should be detected as a category heading"


def test_is_category_heading_still_accepts_real_msft_all_caps_headings():
    real_msft_headings = [
        "STRATEGIC AND COMPETITIVE RISKS",
        "OPERATIONAL RISKS",
        "GENERAL RISKS",
    ]
    for heading in real_msft_headings:
        assert is_category_heading(heading), f"{heading!r} should be detected as a category heading"


def test_is_category_heading_still_rejects_real_msft_sentence_case_subheadings():
    """The critical regression guard: these must NOT be detected as
    category boundaries, or MSFT would be over-fragmented into many
    small categories instead of its real handful."""
    real_msft_subheadings = [
        "Competition in the technology sector",
        "Competition among platform-based ecosystems",
        "Business model competition",
    ]
    for heading in real_msft_subheadings:
        assert not is_category_heading(heading), f"{heading!r} should NOT be detected as a category heading"


def test_is_category_heading_rejects_bare_page_numbers():
    assert not is_category_heading("17")
    assert not is_category_heading("18")


def test_real_oracle_item_1a_title_line_is_dropped_case_insensitively():
    """Second bug found from testing the exact real text (with its
    literal tab character): the ITEM 1A skip check was ALSO ALL-CAPS-only
    (p.startswith("ITEM 1A")), so real Oracle text's Title Case
    "Item 1A.\tRisk Factors" line was never skipped and became its own
    spurious chunk. Same root cause as the main bug, same fix pattern."""
    text_with_real_title_line = "Item 1A.\tRisk Factors\n\nWe operate in rapidly changing environments that present numerous risks and uncertainties.\n\nGeneral Risks\n\nEconomic conditions can adversely affect our business."
    chunks = split_by_category(text_with_real_title_line)

    headings = [c["heading"] for c in chunks]
    assert "Item 1A.\tRisk Factors" not in headings
    assert headings == ["Overview", "General Risks"]
