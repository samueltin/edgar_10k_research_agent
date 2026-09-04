"""Streamlit UI: ticker input, research memo display, and follow-up chat.

Chat is intentionally scoped to only the data already extracted for the
selected company -- it should say so if asked something outside that data,
not guess.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from edgar_research_agent.memo.formatter import generate_memo, check_category_groundedness

st.set_page_config(page_title="10-K Research Agent", layout="wide")
st.title("10-K Research Agent")

ticker = st.text_input("Ticker", value="MSFT").upper().strip()
max_categories_to_summarize = st.number_input(
    "Max risk categories to summarize (0 = summarize none, 999 = summarize all)",
    min_value=0, max_value=999, value=0, step=1,
    help="Limits how many risk categories get sent to the LLM for summarization, in the filing's "
         "own document order. Categories beyond this limit are skipped (no LLM call for them), but "
         "you can still read their original text -- useful for controlling token cost on filings "
         "with many risk categories. 0 skips every named category (Overview is still summarized); "
         "999 is the max, and covers every category any real filing has.",
)

if st.button("Generate memo") and ticker:
    with st.spinner(f"Fetching and analyzing {ticker}..."):
        st.session_state["memo"] = generate_memo(
            ticker, max_categories_to_summarize=max_categories_to_summarize,
        )

memo = st.session_state.get("memo")

if memo:
    st.header(memo["company_name"])

    st.subheader("Key financials")
    kpi_rows = [r.model_dump() for r in memo["kpi_records"]]
    if kpi_rows:
        shared_fields = ["company_name", "company_cik", "fiscal_period", "source"]
        dropped_fields = {"source_location"}
        shared = {
            field: kpi_rows[0][field]
            for field in shared_fields
            if len({row[field] for row in kpi_rows}) == 1
        }
        if shared:
            labels = {
                "company_name": "Company Name",
                "company_cik": "Company CIK",
                "fiscal_period": "Fiscal Period",
                "source": "Source",
            }
            st.markdown(
                " &nbsp;|&nbsp; ".join(
                    f"**{labels[field]}:** {shared[field]}"
                    for field in shared_fields
                    if field in shared
                )
            )
        kpi_rows = [
            {
                k.replace("_", " ").capitalize(): (
                    re.sub(r"(?<!^)(?=[A-Z])", " ", v) if k == "metric_name" else v
                )
                for k, v in row.items()
                if k not in shared and k not in dropped_fields
            }
            for row in kpi_rows
        ]
    st.html("<style>.st-key-kpi_table thead th { text-align: center !important; }</style>")
    with st.container(key="kpi_table"):
        st.table(kpi_rows)

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
    st.table([
        {k.replace("_", " ").capitalize(): v for k, v in s.model_dump().items()}
        for s in memo["segments"]
    ])

    st.subheader("Current Risk factors summary (by category)")
    st.caption(
        "Item 1A, summarized by category -- expand a category to check the summary against its "
        "exact source text. Each summary is groundedness-checked automatically; use Re-check for "
        "a second opinion, since LLM output isn't perfectly deterministic."
    )
    for category in memo["risk_summary_by_category"]:
        with st.expander(f"📌 {category['heading']}"):
            col_summary, col_source = st.columns(2)
            with col_summary:
                if category.get("skipped"):
                    st.caption("LLM summary")
                    st.info(f"⏭️ {category['summary']}")
                else:
                    st.caption("LLM summary")
                    st.markdown(category["summary"])
            with col_source:
                st.caption("Original (as filed)")
                st.text_area(
                    "Source text", category["source_text"],
                    height=300, label_visibility="collapsed",
                    key=f"source_{category['heading']}",
                )

            if category.get("skipped"):
                st.caption("Groundedness not checked -- this category's summary was skipped.")
                continue

            groundedness_key = f"groundedness_{category['heading']}"
            if groundedness_key not in st.session_state:
                st.session_state[groundedness_key] = category["groundedness"]

            if st.button("Re-check groundedness", key=f"recheck_{category['heading']}"):
                with st.spinner("Re-checking summary against source text..."):
                    st.session_state[groundedness_key] = check_category_groundedness(category)

            result = st.session_state[groundedness_key]
            st.caption(f"Backend: {result.backend}")
            if result.grounded:
                st.success("No unsupported claims found.")
            else:
                if result.ungrounded_percentage is not None:
                    st.warning(f"{result.ungrounded_percentage:.0%} of the summary may be unsupported.")
                else:
                    st.warning("Some claims in the summary may not be supported by the source text.")
                for span in result.ungrounded_spans:
                    st.markdown(f"- \u201c{span.text}\u201d")
                    if span.reason:
                        st.caption(span.reason)

    st.subheader("Ask a follow-up question")
    question = st.chat_input("Ask about this company's filing...")
    if question:
        st.chat_message("user").write(question)
        # TODO: wire this to an LLM call scoped to `memo`'s extracted data only
        st.chat_message("assistant").write(
            "Chat wiring not yet implemented -- see memo/formatter.py output "
            "as the only context this should draw from."
        )
