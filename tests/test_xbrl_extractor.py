"""Tests for the XBRL extractor's data-shaping logic.

xbrl_extractor_node is now a graph entry point that fetches its own data
from EDGAR (via get_income_statement), so it can't be unit-tested offline
with a hand-built DataFrame. These tests call the internal
_extract_from_dataframe helper directly instead, which contains all the
actual extraction/normalization logic and has no EDGAR dependency.
"""
import pandas as pd
from edgar_research_agent.agent.nodes.xbrl_extractor import _extract_from_dataframe

SAMPLE_CIK = "0000789019"
SAMPLE_NAME = "Microsoft Corporation"


def _sample_df():
    data = {
        "FY 2023": [211915000000, 100000000000, 72361000000],
        "FY 2022": [198270000000, 90000000000, 72738000000],
    }
    index = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "GrossProfit",
        "NetIncomeLoss",
    ]
    return pd.DataFrame(data, index=index)


def test_extracts_expected_metrics():
    result = _extract_from_dataframe(_sample_df(), SAMPLE_NAME, SAMPLE_CIK)

    metric_names = {r.metric_name for r in result["kpi_records"]}
    assert metric_names == {"Revenue", "GrossProfit", "NetIncome"}


def test_gross_profit_is_not_mislabelled_as_margin():
    result = _extract_from_dataframe(_sample_df(), SAMPLE_NAME, SAMPLE_CIK)

    gross_profit_records = [r for r in result["kpi_records"] if r.metric_name == "GrossProfit"]
    assert len(gross_profit_records) == 2
    assert all(r.unit == "USD" for r in gross_profit_records)


def test_company_name_is_not_the_ticker():
    result = _extract_from_dataframe(_sample_df(), SAMPLE_NAME, SAMPLE_CIK)

    assert all(r.company_name == "Microsoft Corporation" for r in result["kpi_records"])


def test_latest_revenue_used_for_xbrl_total():
    result = _extract_from_dataframe(_sample_df(), SAMPLE_NAME, SAMPLE_CIK)

    assert result["xbrl_total_revenue"] == 211915000000
