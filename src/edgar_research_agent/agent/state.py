"""LangGraph state definition for the research agent pipeline."""
from typing import List, TypedDict
from edgar_research_agent.agent.schemas import KPIRecord, SegmentKPI


class GraphState(TypedDict):
    ticker: str
    company_name: str
    company_cik: str

    kpi_records: List[KPIRecord]        # Track 1 output (XBRL)
    mda_text: str
    risk_factors_text: str
    risk_summary_by_category: List[dict]

    xbrl_total_revenue: float           # ground truth for validation
    extracted_segments: List[SegmentKPI]  # Track 2 output (LLM)

    validation_status: str              # "PASS", "FAIL", "PENDING"
    errors: str
