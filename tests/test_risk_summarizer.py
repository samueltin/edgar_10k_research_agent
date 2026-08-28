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


# --- risk_summarizer_node: max_categories_to_summarize control ---

class _FakeCategorySummary:
    def __init__(self, summary):
        self.summary = summary


class _FakeRiskFactorsSummary:
    def __init__(self, categories):
        self.categories = categories


class _FakeStructuredLLM:
    def __init__(self, categories_to_return):
        self._categories_to_return = categories_to_return

    def invoke(self, prompt):
        return _FakeRiskFactorsSummary(self._categories_to_return)


class _FakeLLM:
    def __init__(self, categories_to_return):
        self._categories_to_return = categories_to_return

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._categories_to_return)


THREE_CATEGORY_TEXT = """ITEM 1A. RISK FACTORS

Intro paragraph before the first real category, long enough to pass the filter.

STRATEGIC AND COMPETITIVE RISKS

We face intense competition across all markets for our products.

OPERATIONAL RISKS

Our business relies on our ability to attract and retain talent.

FINANCIAL RISKS

Our results may fluctuate due to changes in currency exchange rates."""


def test_no_limit_summarizes_every_category(monkeypatch):
    """THREE_CATEGORY_TEXT actually produces 4 chunks: Overview (the
    intro text before the first real heading) plus the 3 named
    categories -- so 4 fake summaries are needed, not 3."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    fake_categories = [
        _FakeCategorySummary("overview summary"),
        _FakeCategorySummary("summary A"),
        _FakeCategorySummary("summary B"),
        _FakeCategorySummary("summary C"),
    ]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM(fake_categories),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT}
    result = risk_summarizer_node(state)

    categories = result["risk_summary_by_category"]
    assert len(categories) == 4
    assert all(c["skipped"] is False for c in categories)
    assert [c["summary"] for c in categories] == ["overview summary", "summary A", "summary B", "summary C"]


def test_limit_to_2_summarizes_first_2_named_categories_and_skips_the_rest(monkeypatch):
    """'Top N' means the first N NAMED categories in the FILING'S OWN
    document order -- there's no other ranking signal in this pipeline.
    Overview (intro boilerplate) doesn't count as one of the N -- see
    test_overview_is_exempt_from_the_limit for why."""
    from edgar_research_agent.agent.nodes.risk_summarizer import (
        risk_summarizer_node, SKIPPED_SUMMARY_MESSAGE,
    )

    fake_categories = [
        _FakeCategorySummary("overview summary"),
        _FakeCategorySummary("summary A"),
        _FakeCategorySummary("summary B"),
    ]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM(fake_categories),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 2}
    result = risk_summarizer_node(state)

    categories = result["risk_summary_by_category"]
    assert len(categories) == 4
    assert categories[0]["heading"] == "Overview"
    assert categories[0]["summary"] == "overview summary"
    assert categories[0]["skipped"] is False
    assert categories[1]["heading"] == "STRATEGIC AND COMPETITIVE RISKS"
    assert categories[1]["summary"] == "summary A"
    assert categories[1]["skipped"] is False
    assert categories[2]["heading"] == "OPERATIONAL RISKS"
    assert categories[2]["summary"] == "summary B"
    assert categories[2]["skipped"] is False
    assert categories[3]["heading"] == "FINANCIAL RISKS"
    assert categories[3]["summary"] == SKIPPED_SUMMARY_MESSAGE
    assert categories[3]["skipped"] is True


def test_overview_is_exempt_from_the_limit(monkeypatch):
    """Real design gap found via testing: Overview is chunk #1 in
    document order, so a naive 'first N chunks' limit would treat it as
    consuming one of the N slots -- meaning 'top 3' would silently
    summarize only 2 REAL named categories. Overview must always be
    summarized (when present) regardless of the limit, and must not
    count against it."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    fake_categories = [
        _FakeCategorySummary("overview summary"),
        _FakeCategorySummary("summary A"),
    ]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM(fake_categories),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 1}
    result = risk_summarizer_node(state)

    categories = result["risk_summary_by_category"]
    overview = next(c for c in categories if c["heading"] == "Overview")
    assert overview["skipped"] is False
    assert overview["summary"] == "overview summary"

    named = [c for c in categories if c["heading"] != "Overview"]
    assert sum(1 for c in named if not c["skipped"]) == 1
    assert sum(1 for c in named if c["skipped"]) == 2


def test_skipped_category_still_keeps_its_real_source_text(monkeypatch):
    """The whole point is saving LLM cost, not hiding the filing's real
    content from the analyst -- a skipped category's original text must
    still be available."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM([_FakeCategorySummary("overview summary"), _FakeCategorySummary("summary A")]),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 1}
    result = risk_summarizer_node(state)

    skipped = result["risk_summary_by_category"][-1]
    assert skipped["skipped"] is True
    assert "currency exchange rates" in skipped["source_text"]


def test_limit_of_zero_means_summarize_none(monkeypatch):
    """0 means summarize none of the named categories -- Overview is still
    exempt and always summarized when present."""
    from edgar_research_agent.agent.nodes.risk_summarizer import (
        risk_summarizer_node, SKIPPED_SUMMARY_MESSAGE,
    )

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM([_FakeCategorySummary("overview summary")]),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 0}
    result = risk_summarizer_node(state)

    categories = result["risk_summary_by_category"]
    overview = next(c for c in categories if c["heading"] == "Overview")
    assert overview["skipped"] is False
    assert overview["summary"] == "overview summary"

    named = [c for c in categories if c["heading"] != "Overview"]
    assert len(named) == 3
    assert all(c["skipped"] for c in named)
    assert all(c["summary"] == SKIPPED_SUMMARY_MESSAGE for c in named)


def test_limit_covering_all_categories_produces_no_skipped_entries(monkeypatch):
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    fake_categories = [
        _FakeCategorySummary("overview summary"),
        _FakeCategorySummary("summary A"),
        _FakeCategorySummary("summary B"),
        _FakeCategorySummary("summary C"),
    ]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM(fake_categories),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 10}
    result = risk_summarizer_node(state)

    assert len(result["risk_summary_by_category"]) == 4
    assert all(not c["skipped"] for c in result["risk_summary_by_category"])


def test_limit_of_999_summarizes_all_categories(monkeypatch):
    """999 is the UI's 'summarize all' value -- real filings have far
    fewer named categories than that, so it must not skip anything."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    fake_categories = [
        _FakeCategorySummary("overview summary"),
        _FakeCategorySummary("summary A"),
        _FakeCategorySummary("summary B"),
        _FakeCategorySummary("summary C"),
    ]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM(fake_categories),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT, "max_categories_to_summarize": 999}
    result = risk_summarizer_node(state)

    assert len(result["risk_summary_by_category"]) == 4
    assert all(not c["skipped"] for c in result["risk_summary_by_category"])


def test_no_chunks_at_all_makes_no_llm_call(monkeypatch):
    """If a filing's Risk Factors text produces zero chunks (e.g. genuinely
    empty text), no LLM call should be made at all -- there's nothing to
    summarize, calling the LLM with an empty chunks_text would be pure
    waste."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    llm_was_called = {"value": False}

    def fake_get_llm():
        llm_was_called["value"] = True
        return _FakeLLM([])

    monkeypatch.setattr("edgar_research_agent.agent.nodes.risk_summarizer.get_llm", fake_get_llm)

    state = {"ticker": "TEST", "risk_factors_text": "", "max_categories_to_summarize": 5}
    result = risk_summarizer_node(state)

    assert result["risk_summary_by_category"] == []
    assert llm_was_called["value"] is False


# --- Provider-aware context limit (context_limits.py) integration ---

def test_uses_provider_aware_context_limit(monkeypatch):
    """Real bug found running llama3.1:8b on real hardware: the old flat
    100,000-char cap (sized for cloud models) badly overran a local
    model's real 4096-token context window, silently truncating the
    prompt. Confirms risk_summarizer_node now asks context_limits.py for
    the right limit rather than using a hardcoded constant."""
    from edgar_research_agent.agent.nodes.risk_summarizer import risk_summarizer_node

    captured = {}

    def fake_get_max_input_chars(task):
        captured["task"] = task
        return 50  # deliberately tiny, to prove it's actually being applied

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_max_input_chars", fake_get_max_input_chars
    )
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.risk_summarizer.get_llm",
        lambda: _FakeLLM([_FakeCategorySummary("summary")]),
    )

    state = {"ticker": "TEST", "risk_factors_text": THREE_CATEGORY_TEXT}
    risk_summarizer_node(state)

    assert captured["task"] == "risk_summarization"
