"""Tests for memo formatting logic -- the derived fields that make
memo/formatter.py an actual formatter, not a pass-through orchestrator."""
from edgar_research_agent.memo.formatter import _compute_gross_margin_pct
from edgar_research_agent.agent.schemas import KPIRecord


def _record(metric_name, year, value):
    return KPIRecord(
        company_cik="0000789019", company_name="Microsoft Corporation",
        fiscal_year=year, fiscal_period="FY", metric_name=metric_name,
        value=value, unit="USD", source="XBRL", source_location=metric_name,
        confidence=1.0,
    )


def test_gross_margin_computed_correctly():
    records = [
        _record("Revenue", 2023, 200_000_000),
        _record("GrossProfit", 2023, 100_000_000),
    ]
    result = _compute_gross_margin_pct(records)
    assert result == {2023: 50.0}


def test_gross_margin_skips_year_with_missing_gross_profit():
    records = [_record("Revenue", 2023, 200_000_000)]
    result = _compute_gross_margin_pct(records)
    assert result == {}


def test_check_category_groundedness_passes_correct_arguments(monkeypatch):
    """The wiring itself is what's being tested here: does the wrapper
    correctly unpack a category dict and pass its fields through to
    check_groundedness -- this is exactly the kind of thing that was
    previously untestable when this call lived directly in
    streamlit_app.py."""
    from edgar_research_agent.memo.formatter import check_category_groundedness

    captured = {}

    def fake_check_groundedness(summary_text, source_text):
        captured["summary_text"] = summary_text
        captured["source_text"] = source_text
        return "fake result"

    monkeypatch.setattr(
        "edgar_research_agent.agent.groundedness.check_groundedness", fake_check_groundedness
    )

    category = {"heading": "General Risks", "summary": "the summary text", "source_text": "the source text"}
    result = check_category_groundedness(category)

    assert result == "fake result"
    assert captured == {"summary_text": "the summary text", "source_text": "the source text"}


def test_generate_memo_merges_groundedness_into_each_category(monkeypatch):
    """generate_memo() now merges the graph's auto-computed
    groundedness_results into each risk_summary_by_category entry, by
    position. This is the wiring that makes the UI able to show a result
    immediately without needing a button click."""
    from edgar_research_agent.agent.groundedness import GroundednessResult
    from edgar_research_agent.memo.formatter import generate_memo

    fake_graph_result = {
        "company_name": "Test Corp",
        "kpi_records": [],
        "extracted_segments": [],
        "validation_status": "PASS",
        "errors": "",
        "mda_text": "",
        "risk_summary_by_category": [
            {"heading": "General Risks", "summary": "summary A", "source_text": "source A"},
            {"heading": "Tax Risks", "summary": "summary B", "source_text": "source B"},
        ],
        "groundedness_results": [
            GroundednessResult(grounded=True, backend="custom_llm"),
            GroundednessResult(grounded=False, backend="custom_llm"),
        ],
    }
    monkeypatch.setattr("edgar_research_agent.agent.graph.run", lambda ticker, max_categories_to_summarize=None: fake_graph_result)

    memo = generate_memo("TEST")

    assert memo["risk_summary_by_category"][0]["heading"] == "General Risks"
    assert memo["risk_summary_by_category"][0]["groundedness"].grounded is True
    assert memo["risk_summary_by_category"][1]["heading"] == "Tax Risks"
    assert memo["risk_summary_by_category"][1]["groundedness"].grounded is False


def test_check_category_groundedness_returns_skipped_result_without_calling_check_groundedness(monkeypatch):
    """Re-checking a skipped category must not waste an LLM call --
    there's no real summary to verify, just a placeholder message."""
    from edgar_research_agent.memo.formatter import check_category_groundedness

    was_called = {"value": False}

    def fake_check_groundedness(summary_text, source_text):
        was_called["value"] = True
        return "should never be returned"

    monkeypatch.setattr(
        "edgar_research_agent.agent.groundedness.check_groundedness", fake_check_groundedness
    )

    category = {
        "heading": "Tax Risks",
        "summary": "Summary skipped due to token usage management.",
        "source_text": "the real source text",
        "skipped": True,
    }
    result = check_category_groundedness(category)

    assert was_called["value"] is False
    assert result.skipped is True
    assert result.backend == "skipped_no_summary"


def test_check_category_groundedness_still_checks_non_skipped_categories(monkeypatch):
    from edgar_research_agent.memo.formatter import check_category_groundedness

    monkeypatch.setattr(
        "edgar_research_agent.agent.groundedness.check_groundedness",
        lambda summary_text, source_text: "real result",
    )

    category = {"heading": "General Risks", "summary": "a real summary", "source_text": "real source", "skipped": False}
    result = check_category_groundedness(category)

    assert result == "real result"


def test_generate_memo_passes_max_categories_to_summarize_through_to_run(monkeypatch):
    """Confirms the control parameter actually reaches agent.graph.run(),
    not just that generate_memo() accepts it."""
    from edgar_research_agent.memo.formatter import generate_memo

    captured = {}

    def fake_run(ticker, max_categories_to_summarize=None):
        captured["ticker"] = ticker
        captured["max_categories_to_summarize"] = max_categories_to_summarize
        return {
            "company_name": "Test Corp", "kpi_records": [], "extracted_segments": [],
            "validation_status": "PASS", "errors": "", "mda_text": "",
            "risk_summary_by_category": [], "groundedness_results": [],
        }

    monkeypatch.setattr("edgar_research_agent.agent.graph.run", fake_run)

    generate_memo("TEST", max_categories_to_summarize=3)

    assert captured["ticker"] == "TEST"
    assert captured["max_categories_to_summarize"] == 3
