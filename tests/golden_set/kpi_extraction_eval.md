# KPI extraction golden set

Following the same evaluation pattern used in `hierarchical_rag_insurance`
(a hand-checked golden test set), this file tracks known-correct extraction
results for a small set of real filings, to catch regressions as the
extraction logic changes.

## Format

For each ticker/fiscal-year, record the expected value pulled by hand from
the actual 10-K, then compare against the pipeline's output.

| Ticker | Fiscal year | Metric | Expected value | Pipeline output | Match? |
|--------|-------------|--------|-----------------|------------------|--------|
| MSFT   | 2023        | Revenue | TBD             | TBD              | TBD    |
| MSFT   | 2023        | GrossProfit | TBD         | TBD              | TBD    |
| AAPL   | 2023        | Revenue | TBD             | TBD              | TBD    |

## Segment extraction spot checks

| Ticker | Fiscal year | Segments (expected) | Segments (extracted) | Validator status |
|--------|-------------|----------------------|------------------------|-------------------|
| MSFT   | 2023        | TBD                  | TBD                    | TBD               |

## Notes

Populate this table by manually reading 3-5 real filings and running the
pipeline against them, once the EDGAR client is working end to end.
