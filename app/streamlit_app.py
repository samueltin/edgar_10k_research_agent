"""Streamlit UI: ticker input, research memo display, and follow-up chat.

Chat is intentionally scoped to only the data already extracted for the
selected company -- it should say so if asked something outside that data,
not guess.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from edgar_research_agent.memo.formatter import generate_memo

st.set_page_config(page_title="10-K Research Agent", layout="wide")
st.title("10-K Research Agent")

ticker = st.text_input("Ticker", value="MSFT").upper().strip()

if st.button("Generate memo") and ticker:
    with st.spinner(f"Fetching and analyzing {ticker}..."):
        st.session_state["memo"] = generate_memo(ticker)

memo = st.session_state.get("memo")

if memo:
    st.header(memo["company_name"])

    st.subheader("Key financials")
    st.table([r.model_dump() for r in memo["kpi_records"]])

    if memo["gross_margin_pct"]:
        st.caption("Gross margin: " + ", ".join(
            f"FY{year}: {pct}%" for year, pct in sorted(memo["gross_margin_pct"].items())
        ))

    st.subheader("Segment performance")
    status = memo["validation_status"]
    if status == "PASS":
        st.success("Validated against XBRL total revenue")
    else:
        st.warning(f"Flagged for review: {memo['errors']}")
    st.table([s.model_dump() for s in memo["segments"]])

    st.subheader("Risk factors")
    st.caption("Item 1A, summarized by category -- expand a category to check the summary against its exact source text.")
    for category in memo["risk_summary_by_category"]:
        with st.expander(f"📌 {category['heading']}"):
            col_summary, col_source = st.columns(2)
            with col_summary:
                st.caption("LLM summary")
                st.markdown(category["summary"])
            with col_source:
                st.caption("Original (as filed)")
                st.text_area(
                    "Source text", category["source_text"],
                    height=300, label_visibility="collapsed",
                    key=f"source_{category['heading']}",
                )

    st.subheader("Ask a follow-up question")
    question = st.chat_input("Ask about this company's filing...")
    if question:
        st.chat_message("user").write(question)
        # TODO: wire this to an LLM call scoped to `memo`'s extracted data only
        st.chat_message("assistant").write(
            "Chat wiring not yet implemented -- see memo/formatter.py output "
            "as the only context this should draw from."
        )
