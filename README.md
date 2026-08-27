# EDGAR 10-K Research Agent

An AI-assisted research co-pilot for equity analysts: pulls structured KPIs and
segment-level data from a company's 10-K filing, validates the numbers, and
presents an analyst-ready memo with a follow-up chat.

## Problem

An analyst building a view on a new company today manually extracts financials,
cross-references prior years, and digs through dense MD&A and footnote text to
find segment detail. This is slow, repetitive, and easy to get wrong.

## What this does

- **Track 1 — deterministic**: pulls revenue, gross profit, and net income
  directly from SEC XBRL company facts. No LLM involved, no hallucination risk.
- **Track 2 — LLM-assisted**: extracts segment-level revenue from MD&A and
  footnote text using structured LLM output.
- **Validation gate**: sums the LLM-extracted segments and compares them
  against the XBRL consolidated total. Mismatches beyond a threshold are
  flagged for human review instead of silently trusted.
- **Memo + chat**: a Streamlit UI renders theoutput as a memo, with a chat layer
  scoped to that company's extracted data only.

Scope: single analyst, single company at a time (not batch screening across a
coverage list — see `docs/architecture.md` for the reasoning).

## Architecture

See `docs/architecture.md` and `docs/c4/10k-research-agent.dsl` for the full
C4 model (Context, Container, Component views).

## Status

Early-stage prototype. Built as a personal project to explore Gen AI use
cases in investment research.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Azure OpenAI / Anthropic credentials
streamlit run app/streamlit_app.py
```

## License

MIT — see LICENSE.
