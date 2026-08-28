# Architecture

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
