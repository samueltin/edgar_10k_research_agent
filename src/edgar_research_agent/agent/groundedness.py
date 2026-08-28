"""Groundedness checking: verifies whether an LLM-generated summary is
actually supported by its source text.

Two backends behind one common interface, selected by GROUNDEDNESS_BACKEND
(env var or explicit argument), default "custom_llm":

- "azure_content_safety": Microsoft's real, managed Groundedness Detection
  API (Azure AI Content Safety, 2024-02-15-preview, Summarization task).
  TESTED AGAINST THE LIVE ENDPOINT: the first real call returned a 400
  Bad Request. Root cause found and fixed: the endpoint URL was built
  without checking for a trailing slash, producing a malformed double-
  slash path Azure's gateway rejected. See _check_azure_content_safety()'s
  docstring for the fix and for a second, still-unverified risk (the
  "reasoning" feature's cross-resource auth), which is now opt-in rather
  than on by default specifically so it can be tested in isolation.
  This backend is Azure-specific: it will not work if the app is
  configured with LLM_PROVIDER=anthropic, since it needs a real Azure
  OpenAI resource for its "reasoning" feature regardless of which
  provider the rest of the app is using.

- "custom_llm": a narrow LLM call through the SAME provider-agnostic
  get_llm() factory used everywhere else in this app (llm_provider.py),
  so it works under either LLM_PROVIDER setting. The LLM quotes the
  specific unsupported claim VERBATIM from the summary, and that quote is
  then checked programmatically against the real summary text before
  being trusted -- the same "never trust an LLM's self-reported quote
  without verifying it" discipline used elsewhere in this project.

Both backends return the same GroundednessResult shape, modeled on
Azure's real response (a boolean, an optional percentage, and a list of
specific unsupported spans with reasons) since that shape is more
informative than a flat pass/fail and the custom backend can produce it
just as naturally.
"""
import os
from typing import List, Optional
from pydantic import BaseModel, Field


class UngroundedSpan(BaseModel):
    text: str = Field(description="The specific unsupported substring, verbatim from the summary")
    reason: str = ""


class GroundednessResult(BaseModel):
    grounded: bool
    ungrounded_percentage: Optional[float] = None  # 0.0-1.0; only ever populated by the Azure backend
    ungrounded_spans: List[UngroundedSpan] = Field(default_factory=list)
    backend: str  # "azure_content_safety", "custom_llm", or "skipped_no_summary"
    skipped: bool = False  # True when the category's summarization itself was skipped (see risk_summarizer.py's
                            # cost-control limit) -- there's no real summary to check, so this was never a genuine
                            # groundedness check, not a check that happened to pass


def check_groundedness(summary_text: str, source_text: str, backend: Optional[str] = None) -> GroundednessResult:
    """Check whether summary_text is grounded in source_text.

    backend defaults to the GROUNDEDNESS_BACKEND env var, or "custom_llm"
    if that isn't set either.
    """
    backend = backend or os.environ.get("GROUNDEDNESS_BACKEND", "custom_llm")

    if backend == "azure_content_safety":
        return _check_azure_content_safety(summary_text, source_text)
    elif backend == "custom_llm":
        return _check_custom_llm(summary_text, source_text)
    else:
        raise ValueError(f"Unknown groundedness backend: {backend!r}")


# --- Azure AI Content Safety backend ---

MAX_RATE_LIMIT_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 2.0


def _check_azure_content_safety(summary_text: str, source_text: str) -> GroundednessResult:
    """Calls Azure AI Content Safety's real Groundedness Detection API.

    REAL BUG FOUND AND FIXED (400 Bad Request on first live test): the
    endpoint URL was built without checking for a trailing slash on the
    endpoint, producing a malformed double-slash path Azure's gateway
    rejected. Fixed by stripping any trailing slash before concatenating.

    SECOND REAL ISSUE FOUND AND FIXED (429 rate limit on second live
    test, AFTER the URL fix): Azure Content Safety's free (F0) pricing
    tier has a very low call-rate limit. Since this node runs one
    sequential call per risk category (see groundedness_checker.py), a
    company with several categories can trip this limit within seconds.
    Fixed with _post_with_retry() below: retries a 429 up to
    MAX_RATE_LIMIT_RETRIES times, honoring Azure's own Retry-After header
    when present.

    HONEST LIMIT OF THIS FIX: retrying smooths over transient throttling,
    it does not raise the F0 tier's actual sustained throughput. If a
    company has many categories, or you run this repeatedly in a short
    period, you may still exhaust the retry budget and fail. Azure's own
    error message says as much: upgrading to a paid Content Safety tier
    is the real fix for sustained use, not more retries.

    THIRD, STILL-UNVERIFIED RISK: `reasoning=True` requires Content
    Safety to call your Azure OpenAI resource on its own behalf. How that
    cross-resource authentication actually works was never confirmed
    against a real call. This remains OPT-IN
    (ENABLE_GROUNDEDNESS_REASONING=true) so it can be tested in isolation
    from the two issues above.
    """
    import requests

    endpoint = os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_CONTENT_SAFETY_KEY"]
    api_version = os.environ.get("AZURE_CONTENT_SAFETY_API_VERSION", "2024-02-15-preview")
    reasoning_enabled = os.environ.get("ENABLE_GROUNDEDNESS_REASONING", "false").lower() == "true"

    payload = {
        "domain": "Generic",
        "task": "Summarization",
        "text": summary_text[:7500],                 # API hard limit, per Microsoft's REST reference
        "groundingSources": [source_text[:55000]],   # API hard limit (55K chars across all sources)
        "reasoning": reasoning_enabled,
    }
    if reasoning_enabled:
        payload["llmResource"] = {
            "resourceType": "AzureOpenAI",
            "azureOpenAIEndpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "azureOpenAIDeploymentName": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        }

    headers = {"Ocp-Apim-Subscription-Key": api_key, "Content-Type": "application/json"}
    url = f"{endpoint}/contentsafety/text:detectGroundedness?api-version={api_version}"

    response = _post_with_retry(requests, url, payload, headers)
    if not response.ok:
        # Azure's actual error body has a specific code/message that
        # response.raise_for_status() discards -- surface it, since that's
        # what actually tells you WHY a request failed, not just that it did.
        raise requests.exceptions.HTTPError(
            f"{response.status_code} error calling Azure Content Safety: {response.text}",
            response=response,
        )
    data = response.json()

    return _parse_azure_response(data)


def _post_with_retry(requests_module, url: str, payload: dict, headers: dict):
    """POST with retry-on-429, honoring Azure's Retry-After header when
    present. requests_module is passed in (rather than imported inside
    this function) so it can be swapped for a test double without any
    monkeypatching gymnastics.
    """
    import time

    response = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        response = requests_module.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 429:
            return response
        if attempt < MAX_RATE_LIMIT_RETRIES - 1:
            time.sleep(_parse_retry_after(response) or DEFAULT_RETRY_DELAY_SECONDS)
    return response  # still 429 after all retries -- caller's .ok check will raise with the real error body


def _parse_retry_after(response) -> Optional[float]:
    header_value = response.headers.get("Retry-After")
    if header_value is None:
        return None
    try:
        return float(header_value)
    except ValueError:
        return None


def _parse_azure_response(data: dict) -> GroundednessResult:
    """Pure mapping from Azure's documented response shape to our common
    schema -- separated from the network call so it can be unit-tested
    with a synthetic response, without needing a real API call.
    """
    spans = [
        UngroundedSpan(text=d.get("text", ""), reason=d.get("reason", ""))
        for d in data.get("ungroundedDetails", [])
    ]
    return GroundednessResult(
        grounded=not data.get("ungroundedDetected", False),
        ungrounded_percentage=data.get("ungroundedPercentage"),
        ungrounded_spans=spans,
        backend="azure_content_safety",
    )


# --- Custom LLM backend ---

CUSTOM_GROUNDEDNESS_PROMPT = """You are checking whether a summary is fully
supported by its source text.

Source text:
{source_text}

Summary to check:
{summary_text}

Identify any claim in the summary that is NOT supported by the source text.
For each one, quote the exact, verbatim substring from the SUMMARY (not the
source) that is unsupported, and give a brief reason. If the summary is
fully supported, return an empty list.
"""


class _CustomGroundednessLLMResult(BaseModel):
    ungrounded_claims: List[UngroundedSpan] = Field(
        default_factory=list,
        description="Exact verbatim substrings from the SUMMARY that are not supported "
                    "by the source text, each with a brief reason. Empty if fully grounded.",
    )


def _check_custom_llm(summary_text: str, source_text: str) -> GroundednessResult:
    """LLM provider import is local so this module stays importable
    without langchain installed, unless this backend is actually used.
    """
    from edgar_research_agent.agent.llm_provider import get_llm

    llm = get_llm()
    structured_llm = llm.with_structured_output(_CustomGroundednessLLMResult).with_retry(
        stop_after_attempt=5, wait_exponential_jitter=True,
    )
    prompt = CUSTOM_GROUNDEDNESS_PROMPT.format(source_text=source_text, summary_text=summary_text)
    result = structured_llm.invoke(prompt)

    verified_spans = _filter_verified_spans(result.ungrounded_claims, summary_text)

    return GroundednessResult(
        grounded=len(verified_spans) == 0,
        ungrounded_percentage=None,  # the custom backend doesn't attempt to estimate this
        ungrounded_spans=verified_spans,
        backend="custom_llm",
    )


def _filter_verified_spans(spans: List[UngroundedSpan], summary_text: str) -> List[UngroundedSpan]:
    """Only trust a flagged span if it's an actual, real substring of the
    summary text -- guards against the LLM hallucinating a quote that
    doesn't really appear anywhere in the summary. Pure function, no LLM
    dependency, fully unit-testable on its own.
    """
    return [span for span in spans if span.text.strip() and span.text.strip() in summary_text]
