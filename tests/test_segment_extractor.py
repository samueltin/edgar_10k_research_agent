"""Tests for segment_extractor_node -- specifically that it uses the
provider-aware context limit rather than a hardcoded constant, added
after a real bug: on real hardware (an 8GB GPU, llama3.1:8b, a confirmed
4096-token context window), the old flat 100,000-character cap silently
truncated the prompt, and the model returned MD&A boilerplate section
headings ("Critical Accounting Estimates", "Recent Accounting Guidance")
as fake business segments, all with 0 revenue -- consistent with the
model only ever seeing the truncated tail of the document.
"""


class _FakeSegment:
    def __init__(self, segment_name, revenue):
        self.segment_name = segment_name
        self.revenue = revenue


class _FakeSegmentExtraction:
    def __init__(self, segments):
        self.segments = segments


class _FakeStructuredLLM:
    def __init__(self, segments_to_return):
        self._segments_to_return = segments_to_return

    def invoke(self, prompt):
        self.last_prompt = prompt
        return _FakeSegmentExtraction(self._segments_to_return)


class _FakeLLM:
    def __init__(self, segments_to_return):
        self._segments_to_return = segments_to_return
        self.structured_llm = None

    def with_structured_output(self, schema):
        self.structured_llm = _FakeStructuredLLM(self._segments_to_return)
        return self.structured_llm


def test_uses_provider_aware_context_limit(monkeypatch):
    """Confirms segment_extractor_node asks context_limits.py for the
    right limit rather than using a hardcoded constant -- the actual
    real-world bug this fixes."""
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    captured = {}

    def fake_get_max_input_chars(task):
        captured["task"] = task
        return 20  # deliberately tiny, to prove it's actually applied to the text

    fake_llm = _FakeLLM([_FakeSegment("Productivity", 1000.0)])

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", fake_get_max_input_chars
    )
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: fake_llm
    )

    long_text = "A" * 1000  # far longer than the 20-char fake limit
    state = {"ticker": "TEST", "mda_text": long_text}
    segment_extractor_node(state)

    assert captured["task"] == "segment_extraction"
    # the prompt actually sent to the LLM should only contain the first
    # 20 characters of mda_text, not all 1000
    assert "A" * 21 not in fake_llm.structured_llm.last_prompt


def test_returns_extracted_segments(monkeypatch):
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    fake_segments = [_FakeSegment("Productivity", 1000.0), _FakeSegment("Cloud", 2000.0)]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", lambda task: 100_000
    )
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: _FakeLLM(fake_segments)
    )

    state = {"ticker": "TEST", "mda_text": "some MD&A text"}
    result = segment_extractor_node(state)

    assert result["extracted_segments"] == fake_segments
    assert result["validation_status"] == "PENDING"
