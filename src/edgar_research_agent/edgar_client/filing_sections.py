"""Fetch narrative sections (MD&A, Risk Factors) from a company's latest 10-K."""
import os
import sys
from edgar import set_identity, Company

set_identity(os.environ.get("EDGAR_IDENTITY", "Your Name <you@example.com>"))


def get_mda_and_risk_factors(ticker: str) -> tuple[str, str]:
    """Return (mda_text, risk_factors_text) from the latest 10-K.

    Item 7 = Management's Discussion and Analysis
    Item 1A = Risk Factors
    """
    company = Company(ticker)
    latest_10k = company.get_filings(form="10-K").latest()
    tenk = latest_10k.obj()

    mda_text = tenk["7"]
    risk_factors_text = tenk["1A"]
    return mda_text, risk_factors_text


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    mda, risks = get_mda_and_risk_factors(ticker)
    print(f"MD&A: {len(mda.split()):,} words")
    print(f"Risk Factors: {len(risks.split()):,} words")
    print(mda)

