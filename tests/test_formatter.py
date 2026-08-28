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
    monkeypatch.setattr("edgar_research_agent.agent.graph.run", lambda ticker: fake_graph_result)

    memo = generate_memo("TEST")

    assert memo["risk_summary_by_category"][0]["heading"] == "General Risks"
    assert memo["risk_summary_by_category"][0]["groundedness"].grounded is True
    assert memo["risk_summary_by_category"][1]["heading"] == "Tax Risks"
    assert memo["risk_summary_by_category"][1]["groundedness"].grounded is False
