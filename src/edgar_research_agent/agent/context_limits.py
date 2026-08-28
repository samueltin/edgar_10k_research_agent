"""Provider-aware input character limits.

MAX_CONTEXT_CHARS was originally a single, provider-blind constant
(100,000 characters, ~25,000 tokens) used by both segment_extractor.py
and risk_summarizer.py. That's a reasonable, well-tested size for
Azure OpenAI/Anthropic's large context windows -- but it silently broke
under Ollama on real hardware: a real llama3.1:8b run on an 8GB GPU
reported a 4096-token context window, meaning roughly 80%+ of a
100,000-character prompt would have been silently truncated before the
model ever saw it. The observed failure (segment_extractor_node
returning MD&A boilerplate section headings -- "Critical Accounting
Estimates", "Recent Accounting Guidance", "Statement of Management's
Responsibility" -- as if they were business segments, all with 0
revenue) is consistent with the model only ever seeing the LATE part of
the document, since truncation drops content from the start.

Rather than just raising OLLAMA_NUM_CTX to match the old 100,000-char
cap, this shrinks the cap itself for Ollama: the same real-hardware test
showed 100% GPU utilization with ZERO demonstrated VRAM headroom even at
a 4096-token window, so pushing num_ctx higher risks spilling the model
off GPU entirely (a different, possibly worse problem) rather than
solving the truncation issue safely.

THESE NUMBERS ARE ESTIMATES, NOT VERIFIED AGAINST REAL OUTPUT QUALITY --
they're sized conservatively from the one real, confirmed data point
(4096 tokens, ~4 chars/token), leaving room for each prompt's own
template text and the model's response, but the actual right numbers
need real testing against your hardware and models, the same discipline
every other fix in this project has been held to. If a real run still
shows truncation or poor quality, these should be reduced further -- if
you raise OLLAMA_NUM_CTX and confirm continued 100% GPU status with
headroom to spare (via `ollama ps`), these can be raised correspondingly.
"""
import os

# Cloud providers (Azure OpenAI, Anthropic) have large, well-tested context
# windows -- unchanged from the original single constant.
CLOUD_MAX_INPUT_CHARS = 100_000

# Ollama: sized conservatively against a REAL confirmed 4096-token window.
# Segment extraction's output is small (a handful of segments), so more of
# the budget can go to input text than risk summarization's, whose output
# (multiple categories x several bullets each) can itself run to
# 500-1500+ tokens, competing with the input for the same context window.
OLLAMA_MAX_INPUT_CHARS_SEGMENT_EXTRACTION = 12_000   # ~3,000 tokens of input, leaving room for template + output
OLLAMA_MAX_INPUT_CHARS_RISK_SUMMARIZATION = 8_000    # ~2,000 tokens of input, leaving more room for larger output


def get_max_input_chars(task: str, provider: str | None = None) -> int:
    """task is "segment_extraction" or "risk_summarization" -- the two
    callers of this function -- since the safe budget differs by how much
    output each task's prompt needs room for, not just by provider.
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "azure_openai")

    if provider != "ollama":
        return CLOUD_MAX_INPUT_CHARS

    if task == "segment_extraction":
        return OLLAMA_MAX_INPUT_CHARS_SEGMENT_EXTRACTION
    elif task == "risk_summarization":
        return OLLAMA_MAX_INPUT_CHARS_RISK_SUMMARIZATION
    else:
        raise ValueError(f"Unknown task: {task!r}")
