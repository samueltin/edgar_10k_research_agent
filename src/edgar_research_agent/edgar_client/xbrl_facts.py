"""Deterministic access to SEC XBRL company facts.

No LLM involved -- this module only fetches and normalizes structured data
that companies already tag in their filings.
"""
import os
from edgar import set_identity, Company

set_identity(os.environ.get("EDGAR_IDENTITY", "Your Name <you@example.com>"))


def get_income_statement(ticker: str):
    """Fetch normalized income statement facts for a ticker as a DataFrame.

    edgartools normalizes the underlying XBRL tags into standard accounting
    concepts (Revenue, GrossProfit, NetIncomeLoss, etc).
    """
    company = Company(ticker)
    facts = company.get_facts()
    income_statement = facts.income_statement()
    return income_statement.to_dataframe(), company.name, str(company.cik)


if __name__ == "__main__":
    df, name, cik = get_income_statement("MSFT")
    print(f"{name} ({cik})")
    print(df.head())
