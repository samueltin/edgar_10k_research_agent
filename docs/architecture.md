# Architecture

## Third round of real bugs: three different root causes from live GOOG/AMZN/ORCL runs

**MSFT passed validation; GOOG, AMZN, and ORCL all failed, each for a
genuinely different reason** -- confirmed by tracing every reported number
back to the real source figures, not assumed from the variance percentage
alone.

**ORCL: adjacent-table bleed, fixed and tested.** The extraction correctly
chose the business-breakdown table over the geography one, but the flat
2,000-character backward window was still wide enough to reach into the
adjacent geography table sitting right next to it (both end in the same
total). The model then reported Americas/EMEA/Asia Pacific as fake
business segments. Fixed by clamping the extraction boundary at the
nearest OTHER candidate's position -- already known from the
disambiguation step, no new detection logic needed. Confirmed on the real
file: the geography table's dollar figures (44,478 / 36,339) are now
excluded while the full business table remains intact.

**ORCL also showed two UNRESOLVED issues**, not fixed by anything above:
a genuine digit misread (44,478 reported as 44,378) and inconsistent unit
scaling (4 of 6 rows short by exactly 1000x, 2 rows correct). These look
like real table-parsing errors from the model itself, not a text-location
problem -- worth watching whether they persist on a re-run, and honestly
possible they're closer to a capability limit than something fixable.

**AMZN: wrong year column, addressed with a prompt instruction (unverified).**
Every reported figure exactly matched the PRIOR year's column, not the
current one -- confirmed by checking that the reported sum matches summing
all three segments' prior-year values precisely. Likely cause: the real
table's year header reads "20242025" with no visible separator. Added an
explicit instruction to use the second (more recent) figure in a two-year
table -- NOT yet confirmed to change model behavior, needs a real re-run.

**GOOG: subtotal rows double-counted as segments, addressed with a prompt
instruction (unverified).** "Google advertising" and "Google Services
total" are subtotals of rows already listed above them, not independent
segments -- summing just those two subtotal rows alone accounts for the
overwhelming majority of the reported ~2.5x overcount. Added an explicit
instruction not to report subtotal/rollup rows as separate segments --
also unverified against a real re-run.

**Honest overall status:** one of four issues found this round
(ORCL's boundary bleed) is deterministic and fully tested. Two (AMZN's
year selection, GOOG's subtotal exclusion) are prompt-level nudges that
need real re-runs to confirm they actually help. One (ORCL's digit
misread and scaling inconsistency) remains unresolved and may reflect a
real limit of this model on dense, multi-column financial tables that a
text-location or prompt fix can't solve.

---

## Segment table location: from a single-company heading guess to a validated, multi-company anchor

**The problem, found on a real MSFT run:** after fixing the context-window
overflow (see below), a run still returned a fabricated number
(214,400,000,000, appearing NOWHERE in the actual source text) for
"Intelligent Cloud", because the real segment revenue table sits
thousands of characters past where segment NAMES are first defined in
MD&A, and when the real number wasn't in view, the model invented one.

**First attempt, and why it was wrong:** searching for MSFT's exact
"SEGMENT RESULTS OF OPERATIONS" heading. Tested against real MD&A text
from four more companies (GOOG, AMZN, ORCL, IBM) before trusting it, and
found it doesn't generalize AT ALL -- only MSFT's lead-in even contains
the word "segment":
- GOOG: "The following table presents revenues by type (in millions):"
- AMZN: "Net sales information is as follows (in millions):"
- ORCL: "Total Revenues by Business:" (closer, but still not "segment")
- IBM: no MD&A content at all -- Item 7 is a ~30-word "incorporated by
  reference" pointer to a separate Annual Report document this pipeline
  doesn't fetch.

**The actual fix, validated against all five real cases:**
`find_segment_table_by_total_revenue()` anchors on the company's own
XBRL-confirmed total revenue figure (Track 1's deterministic ground
truth) instead of guessing at label wording. The real per-segment
breakdown reliably appears immediately before its own total-revenue line
in every company tested.

Two disambiguation signals, both found necessary by testing, not assumed
upfront:
1. **Numeric density** in the ~300 characters immediately preceding each
   candidate match -- a real table has many other large numbers nearby; a
   narrative mention of the total has few or none. Correctly picked the
   right candidate for MSFT, GOOG, and AMZN on its own.
2. **Geography exclusion** -- ORCL separately reports revenue by
   geography, ending in the exact same total, immediately before its real
   by-business breakdown. The two tied on numeric density alone (score 6
   vs 6); excluding candidates preceded by geography-breakdown language
   (a common MD&A pattern generally, not an Oracle-specific keyword)
   broke the tie correctly.

**A real bug found while building this, not just the final design:** an
early version used the SAME small window (300 chars) for both scoring
which candidate is correct AND for the actual extracted text -- this
correctly identified the right position but then returned an incomplete
table (MSFT's result was missing 2 of 3 segments, since the real table is
~1,061 characters, wider than the 300-char scoring window). Fixed by
using a separate, larger window (2,000 characters) for extraction once
the right position is already known.

**IBM needs no special case at all** -- its stub text has no numbers in
it, so the same search naturally finds zero candidates and returns None.
The honest outcome is an empty result the validator correctly flags, not
a fabricated one.

**Validated against real MD&A text from MSFT, GOOG, AMZN, ORCL, and IBM's
stub case.** Falls back to the original capped-full-text behavior when
xbrl_total_revenue is unavailable or no candidate is found in the text,
so a sixth, untested company degrades gracefully rather than breaking.

---

## Documentation gap, worth knowing about

This file (and this repo generally) is missing architecture notes for
several rounds of work already delivered in earlier zips but not yet
pushed to GitHub or merged into your local checkout: Ollama support in
`llm_provider.py`, the groundedness backend's Azure bug fixes (URL
double-slash, 429 retry logic), and the `max_categories_to_summarize`
cost control. All of that code IS present in this delivery (reapplied
from those earlier zips), but this doc file only covers what's new in
THIS round below, not the full history. Worth reconciling your actual
running setup, your local files, and GitHub into one consistent state
when you get a chance -- right now there are at least three different
versions of this codebase in play.

## Real bug: local model context truncation silently corrupted segment extraction

**What happened:** running `llama3.1:8b` via Ollama on real hardware (an
8GB GPU), `segment_extractor_node` returned a list of "segments" that
were actually real MD&A boilerplate section headings -- `cash_flows`,
`share_repurchases`, `dividends`, `critical_accounting_estimates`,
`recent_accounting_guidance`, `statement_of_management_s_responsibility_
for_financial_statements` -- all with 0 revenue. None of these are
business segments; they're standard Item 7 topics that appear in nearly
every 10-K.

**Root cause, confirmed with real data:** `segment_extractor_node` and
`risk_summarizer_node` both capped their input at a flat 100,000
characters (~25,000 tokens), sized for cloud models' large context
windows. A real `ollama run llama3.1` on the actual hardware reported a
**4096-token context window**, already at 100% GPU utilization with no
demonstrated VRAM headroom. The ~25,000-token prompt was almost entirely
truncated before the model ever saw it. Since truncation drops content
from the start, and the real segment-revenue table typically appears
EARLY in MD&A while boilerplate items appear LATE, the model very likely
never saw the real segment data at all -- only the tail end of the
document, which is exactly the boilerplate content it reported back.

**Why the fix isn't "raise OLLAMA_NUM_CTX higher":** the same real test
showed 100% GPU utilization already, with no spare capacity demonstrated.
Pushing the context window higher risks spilling the model off GPU
entirely -- a different, possibly worse failure. The safer fix is
shrinking what gets sent to the model, not expanding the window to match
an oversized prompt.

**The fix:** `context_limits.py`, a provider- and task-aware character
cap. Cloud providers (Azure OpenAI, Anthropic) keep the original
100,000-character limit, already proven to work. Ollama gets a much
smaller limit per task -- 12,000 characters for segment extraction (small
output, more budget for input), 8,000 for risk summarization (potentially
much larger output across many categories, so less budget left for
input). **These numbers are estimates**, sized conservatively from the
one real, confirmed data point (4096 tokens), not verified against real
output quality yet -- if a real run still shows truncation or poor
quality, they should be reduced further; if you confirm headroom to raise
OLLAMA_NUM_CTX safely (via `ollama ps` showing continued 100% GPU status),
they can be raised correspondingly.

**Not yet built, worth considering next:** a more robust fix than a
character-count guess would pre-filter the MD&A text down to just the
segment-revenue subsection before it reaches the LLM at all -- the same
architectural principle `risk_summarizer.py` already uses successfully
for Item 1A (split by real document structure first, then send the LLM
only the relevant slice). This would need real MD&A text samples to
design and test against, the same discipline every other structural fix
in this project has required -- not built speculatively without that.

---

## Scope decision: single analyst, single company (Option A)

This prototype is scoped to one analyst working on one company at a time,
not batch processing or cross-company screening. That decision drives most
of the choices below. See the C4 model in `c4/10k-research-agent.dsl` for
the full Context / Container / Component views.

## Why no vector database (yet)

A vector database solves retrieval across many documents. At this scope,
there is no "haystack" -- a single filing's relevant sections (MD&A, Risk
Factors) fit comfortably in the LLM's context window. Adding a vector store
now would add embedding, chunking, and index-management complexity without
a real retrieval problem to solve.

It becomes worth adding if the project extends to multi-year comparison
across many filings, or cross-company search (Option B).

## Why the EDGAR client is a separate container

`edgar_client/` is deliberately isolated from `agent/`, even though nothing
today requires it. It's a candidate for wrapping as an MCP server later --
so that other AI clients (not just this agent) could call the same SEC
EDGAR access tools. Keeping the boundary clean now makes that a swap of the
container's internals later, not a redesign.

## Why the LLM provider is behind a factory

`agent/llm_provider.py` returns a LangChain-compatible chat model based on
config, rather than importing `AzureChatOpenAI` directly in the graph nodes.
This means swapping from Azure OpenAI to Claude (or another provider) is a
one-line config change, not a rewrite -- avoiding vendor lock-in.

## Why there's a validation gate, not just extraction

Track 2 (LLM segment extraction) is checked against Track 1 (deterministic
XBRL totals) before being trusted. A mismatch beyond a relative threshold
routes to a "flagged for human review" state rather than being silently
accepted. This is the core design principle of the whole project: AI speeds
up preparation, but a human stays in the loop for anything that doesn't
reconcile.
