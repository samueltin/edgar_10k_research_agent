"""Pydantic schemas shared across the extraction and validation pipeline."""
from typing import List
from pydantic import BaseModel, Field


class KPIRecord(BaseModel):
    """A single structured financial data point, sourced from XBRL or an LLM."""
    company_cik: str
    company_name: str
    fiscal_year: int
    fiscal_period: str          # "FY", "Q1", "Q2", ...
    metric_name: str            # "Revenue", "GrossProfit", "NetIncome"
    value: float
    unit: str                   # "USD", "%"
    source: str                 # "XBRL" or "LLM_extracted"
    source_location: str        # XBRL concept tag, or section reference
    confidence: float           # 1.0 for XBRL, LLM confidence score otherwise


class SegmentKPI(BaseModel):
    """A single business segment's reported revenue, as extracted by the LLM."""
    segment_name: str = Field(description="The name of the business operating segment")
    revenue: float = Field(description="The revenue for this segment in USD")


class SegmentExtraction(BaseModel):
    """LLM structured-output container for all segments found in one filing."""
    segments: List[SegmentKPI] = Field(description="All extracted business segments")
