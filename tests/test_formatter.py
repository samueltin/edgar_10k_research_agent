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
