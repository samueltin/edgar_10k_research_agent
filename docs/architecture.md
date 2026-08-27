# Architecture

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
