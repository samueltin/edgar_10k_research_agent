"""Component: XBRL extractor.

Deterministic, no LLM. Pulls target financial concepts from the XBRL
company-facts DataFrame and normalizes them into KPIRecord objects.
"""
import pandas as pd
from edgar_research_agent.agent.schemas import KPIRecord
from edgar_research_agent.agent.state import GraphState

# XBRL concept -> our metric name. Note: GrossProfit is a dollar figure,
# not a margin percentage -- keep the naming accurate.
TARGET_CONCEPTS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "GrossProfit": "GrossProfit",
    "NetIncomeLoss": "NetIncome",
}


def xbrl_extractor_node(state: GraphState) -> dict:
    """Graph entry point. Fetches the XBRL DataFrame itself (using
    state["ticker"]) so xbrl_total_revenue is produced inside the graph,
    not seeded into state before graph.invoke() is called.

    The edgartools-dependent import is local to this function, not at
    module level, so _extract_from_dataframe (all the actual logic) can be
    unit-tested without edgartools installed.
    """
    from edgar_research_agent.edgar_client.xbrl_facts import get_income_statement

    df, company_name, cik = get_income_statement(state["ticker"])
    return _extract_from_dataframe(df, company_name, cik)


def _extract_from_dataframe(df: pd.DataFrame, company_name: str, company_cik: str) -> dict:
    records: list[KPIRecord] = []
    skipped: list[str] = []

    df_filtered = df[df.index.isin(TARGET_CONCEPTS.keys())]

    for concept, row in df_filtered.iterrows():
        metric_name = TARGET_CONCEPTS[concept]

        for col in df.columns:
            if not str(col).startswith("FY "):
                continue
            year = int(str(col).replace("FY ", ""))
            value = row[col]

            if pd.isna(value):
                skipped.append(f"{concept} {year}")
                continue

            records.append(KPIRecord(
                company_cik=company_cik,
                company_name=company_name,
                fiscal_year=year,
                fiscal_period="FY",
                metric_name=metric_name,
                value=float(value),
                unit="USD",
                source="XBRL",
                source_location=concept,
                confidence=1.0,
            ))

    if skipped:
        print(f"XBRL extractor: skipped {len(skipped)} missing values: {skipped}")

    latest_revenue = next(
        (r.value for r in sorted(records, key=lambda r: r.fiscal_year, reverse=True)
         if r.metric_name == "Revenue"),
        0.0,
    )

    return {
        "kpi_records": records,
        "xbrl_total_revenue": latest_revenue,
        "company_name": company_name,
        "company_cik": company_cik,
    }
