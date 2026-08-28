"""Tests for groundedness.py.

_filter_verified_spans() and _parse_azure_response() are pure functions
with no LLM or network dependency and are tested directly. The two
backend entry points (_check_custom_llm, _check_azure_content_safety)
are tested via monkeypatching, the same pattern used elsewhere in this
project for LLM- and network-dependent code.
"""
from edgar_research_agent.agent.groundedness import (
    check_groundedness,
    UngroundedSpan,
    GroundednessResult,
    _filter_verified_spans,
    _parse_azure_response,
)


# --- Dispatch logic ---

def test_check_groundedness_raises_on_unknown_backend():
    try:
        check_groundedness("summary", "source", backend="not_a_real_backend")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_check_groundedness_dispatches_to_custom_llm_by_default(monkeypatch):
    monkeypatch.delenv("GROUNDEDNESS_BACKEND", raising=False)
    called = {}

    def fake_custom_llm(summary_text, source_text):
        called["summary_text"] = summary_text
        called["source_text"] = source_text
        return GroundednessResult(grounded=True, backend="custom_llm")

    monkeypatch.setattr("edgar_research_agent.agent.groundedness._check_custom_llm", fake_custom_llm)

    result = check_groundedness("a summary", "a source")

    assert result.backend == "custom_llm"
    assert called == {"summary_text": "a summary", "source_text": "a source"}


def test_check_groundedness_dispatches_via_env_var(monkeypatch):
    monkeypatch.setenv("GROUNDEDNESS_BACKEND", "azure_content_safety")
    monkeypatch.setattr(
        "edgar_research_agent.agent.groundedness._check_azure_content_safety",
        lambda summary_text, source_text: GroundednessResult(grounded=True, backend="azure_content_safety"),
    )

    result = check_groundedness("summary", "source")

    assert result.backend == "azure_content_safety"


def test_explicit_backend_argument_overrides_env_var(monkeypatch):
    monkeypatch.setenv("GROUNDEDNESS_BACKEND", "azure_content_safety")
    monkeypatch.setattr(
        "edgar_research_agent.agent.groundedness._check_custom_llm",
        lambda summary_text, source_text: GroundednessResult(grounded=True, backend="custom_llm"),
    )

    result = check_groundedness("summary", "source", backend="custom_llm")

    assert result.backend == "custom_llm"


# --- Custom LLM backend: quote verification (pure, deterministic) ---

def test_filter_verified_spans_keeps_real_substring():
    spans = [UngroundedSpan(text="a claim that really appears", reason="not in source")]
    summary = "This summary contains a claim that really appears in it."

    verified = _filter_verified_spans(spans, summary)

    assert len(verified) == 1


def test_filter_verified_spans_drops_hallucinated_quote():
    """The core safety guard: if the LLM claims to quote the summary but
    the quoted text doesn't actually appear there, it must be dropped,
    not trusted."""
    spans = [UngroundedSpan(text="this text was never actually written", reason="hallucinated")]
    summary = "This is the real summary text, completely different from the claim above."

    verified = _filter_verified_spans(spans, summary)

    assert verified == []


def test_filter_verified_spans_drops_empty_text():
    spans = [UngroundedSpan(text="   ", reason="empty")]
    summary = "Some summary text."

    verified = _filter_verified_spans(spans, summary)

    assert verified == []


def test_filter_verified_spans_keeps_only_the_real_ones_from_a_mixed_list():
    spans = [
        UngroundedSpan(text="real claim in the summary", reason="a"),
        UngroundedSpan(text="fabricated claim not present", reason="b"),
    ]
    summary = "This has a real claim in the summary, nothing else."

    verified = _filter_verified_spans(spans, summary)

    assert len(verified) == 1
    assert verified[0].text == "real claim in the summary"


def test_check_custom_llm_end_to_end_via_monkeypatched_llm(monkeypatch):
    """Exercises _check_custom_llm itself (not just the dispatcher),
    confirming the hallucination filter is actually wired into the real
    call path, not just tested in isolation."""
    from edgar_research_agent.agent.groundedness import _CustomGroundednessLLMResult

    class FakeStructuredLLM:
        def with_retry(self, **kwargs):
            return self

        def invoke(self, prompt):
            return _CustomGroundednessLLMResult(ungrounded_claims=[
                UngroundedSpan(text="a real unsupported claim", reason="not in source"),
                UngroundedSpan(text="a made up quote", reason="hallucinated, not really in summary"),
            ])

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    monkeypatch.setattr(
        "edgar_research_agent.agent.llm_provider.get_llm",
        lambda *args, **kwargs: FakeLLM(),
    )

    from edgar_research_agent.agent.groundedness import _check_custom_llm

    summary = "This summary contains a real unsupported claim, and nothing else questionable."
    result = _check_custom_llm(summary, "some source text")

    assert result.backend == "custom_llm"
    assert result.grounded is False
    assert len(result.ungrounded_spans) == 1
    assert result.ungrounded_spans[0].text == "a real unsupported claim"


# --- Azure backend: response parsing (pure, using Microsoft's documented example shape) ---

def test_parse_azure_response_fully_grounded():
    data = {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    result = _parse_azure_response(data)

    assert result.grounded is True
    assert result.ungrounded_percentage == 0.0
    assert result.ungrounded_spans == []
    assert result.backend == "azure_content_safety"


def test_parse_azure_response_with_ungrounded_details():
    """Uses the exact example response shape from Microsoft's published
    REST reference for this API."""
    data = {
        "ungroundedDetected": True,
        "ungroundedPercentage": 0.3,
        "ungroundedDetails": [
            {
                "text": "The sun rises from the west.",
                "offset": {"utf8": 0, "utf16": 0, "codePoint": 0},
                "length": {"utf8": 28, "utf16": 28, "codePoint": 28},
                "reason": "The sun rises from the east due to the visual effect caused by the Earth",
            }
        ],
    }

    result = _parse_azure_response(data)

    assert result.grounded is False
    assert result.ungrounded_percentage == 0.3
    assert len(result.ungrounded_spans) == 1
    assert result.ungrounded_spans[0].text == "The sun rises from the west."
    assert "visual effect" in result.ungrounded_spans[0].reason


def test_check_azure_content_safety_via_monkeypatched_requests(monkeypatch):
    """Exercises _check_azure_content_safety itself, confirming the
    request is built correctly and the response is parsed via
    _parse_azure_response -- without a real network call."""
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    result = _check_azure_content_safety("a summary", "a source")

    assert result.grounded is True
    assert result.backend == "azure_content_safety"
    assert captured["json"]["task"] == "Summarization"
    assert captured["json"]["text"] == "a summary"
    assert captured["json"]["groundingSources"] == ["a source"]
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "fake-key"
    assert "detectGroundedness" in captured["url"]


def test_no_double_slash_when_endpoint_has_trailing_slash(monkeypatch):
    """Regression test for a real 400 Bad Request found on the first live
    test: AZURE_CONTENT_SAFETY_ENDPOINT with a trailing slash produced a
    double-slash URL (".../azure.com//contentsafety/...") that Azure's
    API gateway rejected outright."""
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    _check_azure_content_safety("a summary", "a source")

    assert "//contentsafety" not in captured["url"]
    assert captured["url"] == "https://fake.cognitiveservices.azure.com/contentsafety/text:detectGroundedness?api-version=2024-02-15-preview"


def test_reasoning_is_off_by_default(monkeypatch):
    """The cross-resource auth needed for 'reasoning' was never verified
    against a real call -- it must stay opt-in, not on by default."""
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    _check_azure_content_safety("a summary", "a source")

    assert captured["json"]["reasoning"] is False
    assert "llmResource" not in captured["json"]


def test_reasoning_enabled_via_env_var_includes_llm_resource(monkeypatch):
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.setenv("ENABLE_GROUNDEDNESS_REASONING", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-openai.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "fake-deployment")

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    _check_azure_content_safety("a summary", "a source")

    assert captured["json"]["reasoning"] is True
    assert captured["json"]["llmResource"]["azureOpenAIEndpoint"] == "https://fake-openai.openai.azure.com"


def test_http_error_surfaces_response_body(monkeypatch):
    """A bare raise_for_status() only gives a generic '400 Bad Request'
    message and discards Azure's actual error body -- the real error text
    is what actually explains WHY a request failed. This must be
    surfaced, not swallowed."""
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    class FakeResponse:
        ok = False
        status_code = 400
        text = '{"error": {"code": "InvalidRequestBody", "message": "groundingSources is required"}}'

    def fake_post(url, json, headers, timeout):
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    try:
        _check_azure_content_safety("a summary", "a source")
        assert False, "expected an HTTPError"
    except requests.exceptions.HTTPError as e:
        assert "groundingSources is required" in str(e)


# --- Rate-limit retry: real 429 found on second live test, after the URL fix ---

def test_retries_on_429_and_succeeds(monkeypatch):
    """Regression test for a real 429 hit on live Azure Content Safety
    (F0 free tier rate limit). Simulates two 429s then a success."""
    import time
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    call_count = {"n": 0}

    class RateLimitedResponse:
        ok = False
        status_code = 429
        headers = {}

    class SuccessResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return RateLimitedResponse()
        return SuccessResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    result = _check_azure_content_safety("a summary", "a source")

    assert result.grounded is True
    assert call_count["n"] == 3  # 2 failed attempts + 1 success
    assert len(sleep_calls) == 2  # slept before each retry, not after the final success


def test_honors_retry_after_header(monkeypatch):
    """Azure's real 429 response can include a Retry-After header --
    honor it instead of always using the fixed default delay."""
    import time
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    call_count = {"n": 0}

    class RateLimitedResponse:
        ok = False
        status_code = 429
        headers = {"Retry-After": "5"}

    class SuccessResponse:
        ok = True
        status_code = 200

        def json(self):
            return {"ungroundedDetected": False, "ungroundedPercentage": 0.0, "ungroundedDetails": []}

    def fake_post(url, json, headers, timeout):
        call_count["n"] += 1
        return RateLimitedResponse() if call_count["n"] == 1 else SuccessResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    _check_azure_content_safety("a summary", "a source")

    assert sleep_calls == [5.0]  # used the real header value, not the 2.0s default


def test_gives_up_after_max_retries_and_raises_with_real_error_body(monkeypatch):
    """If the rate limit never clears, this must fail loudly with Azure's
    real error message, not hang forever or fail silently."""
    import time
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    monkeypatch.setenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_SAFETY_KEY", "fake-key")
    monkeypatch.delenv("ENABLE_GROUNDEDNESS_REASONING", raising=False)

    class AlwaysRateLimited:
        ok = False
        status_code = 429
        headers = {}
        text = '{"error":{"code":"429","message":"rate limit exceeded, upgrade your tier"}}'

    def fake_post(url, json, headers, timeout):
        return AlwaysRateLimited()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    from edgar_research_agent.agent.groundedness import _check_azure_content_safety

    try:
        _check_azure_content_safety("a summary", "a source")
        assert False, "expected an HTTPError after exhausting retries"
    except requests.exceptions.HTTPError as e:
        assert "upgrade your tier" in str(e)


def test_parse_retry_after_missing_header_returns_none():
    from edgar_research_agent.agent.groundedness import _parse_retry_after

    class Response:
        headers = {}

    assert _parse_retry_after(Response()) is None


def test_parse_retry_after_invalid_header_returns_none():
    from edgar_research_agent.agent.groundedness import _parse_retry_after

    class Response:
        headers = {"Retry-After": "not-a-number"}

    assert _parse_retry_after(Response()) is None
