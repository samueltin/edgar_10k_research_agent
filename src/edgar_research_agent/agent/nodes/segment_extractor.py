"""Component: Segment extractor.

Uses an LLM with structured output to extract segment-level revenue from
MD&A / footnote text. Deliberately does NOT instruct the model to force its
numbers to reconcile with the total -- that would undermine the validator's
job of catching real discrepancies honestly.

REAL BUG FOUND running llama3.1:8b on real hardware (an 8GB GPU): the input
text was capped at a flat 100,000 characters regardless of provider, sized
for cloud models' large context windows. Ollama's real, confirmed context
window on that hardware was only 4096 tokens -- the prompt was silently
truncated. Fixed by using a provider- and task-aware limit (see
context_limits.py) instead of a single constant.

SECOND REAL BUG: raising the cap alone wasn't enough. A real MSFT run still
returned a fabricated number (214,400,000,000, appearing NOWHERE in the
actual source text) for "Intelligent Cloud", because the real segment table
sits thousands of characters past where segment NAMES are first defined,
and when the real number wasn't in view, the model invented one rather than
reporting it lacked the data.

An initial fix searched for MSFT's exact "SEGMENT RESULTS OF OPERATIONS"
heading -- but real MD&A text from four more companies (GOOG, AMZN, ORCL,
IBM) showed this doesn't generalize AT ALL: only MSFT's lead-in even
contains the word "segment". GOOG says "revenues by type", AMZN says "Net
sales information", ORCL says "Total Revenues by Business" -- three
different conventions, none matching each other or MSFT.

THE ACTUAL FIX: rather than guess at company-specific label wording,
anchor on something we already know is correct -- the company's real,
XBRL-confirmed total revenue figure (Track 1's deterministic ground
truth). find_segment_table_by_total_revenue() searches for that number
(formatted in millions) in the MD&A text; the real per-segment breakdown
table reliably appears immediately before its own total-revenue line in
every one of the four real companies tested. Disambiguates using two
signals FOUND BY TESTING, not assumed upfront: (1) prefer whichever
occurrence has more other large numbers immediately preceding it (a real
table has many; a narrative mention of the total has few or none), and
(2) exclude candidates preceded by geography-breakdown language (a very
common MD&A pattern -- e.g. ORCL separately reports revenue by geography,
ending in the SAME total, right before its real by-business breakdown --
without this exclusion the two tie on the numeric-density signal alone).

IBM is handled by the same mechanism with no special case needed: its
Item 7 is just a ~30-word "incorporated by reference" pointer with no
numbers in it at all, so the total-revenue search naturally finds zero
candidates and returns None -- there is genuinely no segment data in the
10-K itself to extract, and the honest outcome is an empty result the
validator correctly flags, not a fabricated one.

Validated against real MD&A text from MSFT, GOOG, AMZN, ORCL, and IBM's
stub case -- not assumed to generalize further than that. Falls back to
the old capped-full-text behavior when xbrl_total_revenue is unavailable
or no candidate is found, so an unmatched company degrades gracefully
rather than breaking.

THIRD ROUND OF REAL BUGS, found from actual live runs against MSFT
(passed), GOOG, AMZN, and ORCL (all three flagged FAIL by the validator,
correctly, with three DIFFERENT root causes, not one bug repeated):

- ORCL: the extraction correctly chose the business-breakdown table over
  the geography one, but the flat 2,000-char backward window was still
  wide enough to bleed backward into the adjacent geography table sitting
  right next to it (both tables end in the same total). Fixed by clamping
  the extraction boundary at the nearest OTHER candidate's position --
  already known from the disambiguation step, no new detection needed.
  ORCL also showed a separate, still-unresolved digit misread (44,478
  reported as 44,378) and inconsistent unit scaling (4 of 6 rows off by
  exactly 1000x) -- neither is fixed by anything in this round; they
  looks like genuine table-parsing errors from the model itself.

- AMZN: every reported figure exactly matched the PRIOR year's column,
  not the current year's -- confirmed by checking that the reported sum
  matches summing all three segments' prior-year values precisely. Likely
  cause: the real table's year header reads "20242025" with no visible
  separator, and each row lists two values back-to-back distinguished
  only by a non-breaking space -- a genuinely ambiguous format. Addressed
  with an explicit prompt instruction to use the second (more recent)
  figure in a two-year table.

- GOOG: the model included SUBTOTAL rows ("Google advertising", "Google
  Services total") as if they were independent segments, on top of the
  individual line items that sum to them -- explaining the reported total
  running to nearly 2.5x the real XBRL figure. Addressed with an explicit
  prompt instruction not to report subtotal/rollup rows as separate
  segments.

HONEST STATUS: the ORCL boundary-clip fix is deterministic and tested
directly (see tests). The AMZN and GOOG prompt instructions are
reasonable, evidence-informed additions, NOT verified to actually change
model behavior -- they need a real re-run to confirm, the same as every
other prompt change in this project. ORCL's digit misread and scaling
inconsistency remain unresolved; they may be a genuine capability limit
of this model on dense, multi-column financial tables rather than
something a location or prompt fix can address.
"""
import re
from edgar_research_agent.agent.schemas import SegmentExtraction
from edgar_research_agent.agent.state import GraphState
from edgar_research_agent.agent.llm_provider import get_llm
from edgar_research_agent.agent.context_limits import get_max_input_chars

# Small window for DISAMBIGUATION scoring (which candidate is real), kept
# tight deliberately -- a wider window bleeds into an ADJACENT table (found
# with real Oracle text: an 800-char lookback from the real "by business"
# table's total reached backward into the separate "by geography" table
# immediately before it, wrongly flagging BOTH as geographic). The real
# distance from a table's own heading to its own total was measured at
# ~250-265 characters across real Oracle tables; 300 gives a small margin.
SCORE_WINDOW_CHARS = 300

# Separate, much larger window for the actual EXTRACTED text once the right
# position is chosen -- the real MSFT table alone needs ~1,042 characters
# for all three segments; 2000 gives comfortable margin for companies with
# more segments or more verbose formatting.
EXTRACTION_WINDOW_CHARS = 2000

GEOGRAPHY_PATTERN = re.compile(r"geography|geographic|americas|EMEA|Asia Pacific", re.IGNORECASE)
LARGE_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+")

PROMPT_TEMPLATE = """You are an expert financial analyst. Extract the
segment-level revenue for {ticker} from the following MD&A / footnote text.
Report the segments exactly as described in the text, with their reported
revenue figures. Do not adjust figures to make them sum to any particular
total -- report what the text actually states.

Financial tables are commonly presented "in millions" or "in thousands" --
check the table header or surrounding text for this scale indicator, and
report each segment's revenue as a full USD amount (e.g. a table stated "in
millions" showing 139,996 must be reported as 139996000000).

If the text does not contain an actual reported revenue figure for a
segment, do not estimate, guess, or recall a number from general knowledge
-- omit that segment entirely rather than inventing a figure.

Many tables show TWO years of figures side by side for the same segment
(e.g. a prior year followed by the current year). Use only the MOST RECENT
year's figure for each segment -- in most 10-K tables, this is the SECOND
number listed for each row, not the first. Check the year labels in the
table header carefully to confirm which number is the current year before
choosing.

Some tables include SUBTOTAL or rollup rows that are the SUM of other rows
already listed (for example, a combined total for two product lines listed
separately above it). Do not report a subtotal row as if it were its own
separate segment -- only report the individual, non-overlapping base-level
segments, not any row that sums other rows in the same table.

Text:
{mda_text}
"""


def find_segment_table_by_total_revenue(mda_text: str, xbrl_total_revenue: float) -> str | None:
    """Locate the real segment revenue table by anchoring on the company's
    own known total revenue (from XBRL), rather than guessing at
    company-specific heading wording. Returns None if xbrl_total_revenue
    is falsy or no occurrence of the formatted total is found in the text
    -- callers should fall back to the original full (capped) text in
    that case.
    """
    if not xbrl_total_revenue:
        return None

    formatted_total = f"{xbrl_total_revenue / 1_000_000:,.0f}"

    candidates = []
    idx = 0
    while True:
        idx = mda_text.find(formatted_total, idx)
        if idx == -1:
            break
        preceding_window = mda_text[max(0, idx - SCORE_WINDOW_CHARS):idx]
        candidates.append({
            "position": idx,
            "score": len(LARGE_NUMBER_PATTERN.findall(preceding_window)),
            "is_geographic": bool(GEOGRAPHY_PATTERN.search(preceding_window)),
        })
        idx += 1

    if not candidates:
        return None

    non_geographic = [c for c in candidates if not c["is_geographic"]]
    pool = non_geographic if non_geographic else candidates
    best = max(pool, key=lambda c: c["score"])

    # REAL BUG FOUND on a real ORCL run: even after correctly choosing the
    # business-breakdown candidate over the geography-breakdown one, the
    # flat 2,000-char backward window was wide enough to bleed BACKWARD
    # into the adjacent geography table anyway (both tables sit close
    # together, ending in the same total). The model then reported
    # Americas/EMEA/Asia Pacific as if they were business segments.
    # Fixed by clamping the extraction boundary at the nearest OTHER
    # candidate's position -- we already know where it is from the
    # disambiguation step above, no extra detection needed.
    other_before = [c["position"] for c in candidates if c["position"] < best["position"]]
    nearest_other_before = max(other_before, default=None)

    natural_start = max(0, best["position"] - EXTRACTION_WINDOW_CHARS)
    extraction_start = natural_start
    if nearest_other_before is not None:
        extraction_start = max(natural_start, min(nearest_other_before + len(formatted_total), best["position"]))

    return mda_text[extraction_start:best["position"] + len(formatted_total) + 60]


def segment_extractor_node(state: GraphState) -> dict:
    max_chars = get_max_input_chars("segment_extraction")
    mda_text = state["mda_text"]

    segment_table_text = find_segment_table_by_total_revenue(mda_text, state.get("xbrl_total_revenue"))
    text_to_send = segment_table_text if segment_table_text is not None else mda_text

    llm = get_llm()
    structured_llm = llm.with_structured_output(SegmentExtraction)

    prompt = PROMPT_TEMPLATE.format(
        ticker=state["ticker"],
        mda_text=text_to_send[:max_chars],
    )

    result = structured_llm.invoke(prompt)
    return {"extracted_segments": result.segments, "validation_status": "PENDING"}
