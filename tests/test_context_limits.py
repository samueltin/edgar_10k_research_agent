"""Tests for context_limits.py -- the provider- and task-aware input
character cap, added after a real bug: segment_extractor_node and
risk_summarizer_node both used a flat 100,000-character cap sized for
cloud models, which badly overran a real, confirmed 4096-token Ollama
context window on real hardware.
"""
from edgar_research_agent.agent.context_limits import (
    get_max_input_chars,
    CLOUD_MAX_INPUT_CHARS,
    OLLAMA_MAX_INPUT_CHARS_SEGMENT_EXTRACTION,
    OLLAMA_MAX_INPUT_CHARS_RISK_SUMMARIZATION,
)


def test_cloud_providers_keep_the_original_large_limit(monkeypatch):
    assert get_max_input_chars("segment_extraction", provider="azure_openai") == CLOUD_MAX_INPUT_CHARS
    assert get_max_input_chars("risk_summarization", provider="anthropic") == CLOUD_MAX_INPUT_CHARS


def test_ollama_gets_a_much_smaller_limit_per_task():
    assert get_max_input_chars("segment_extraction", provider="ollama") == OLLAMA_MAX_INPUT_CHARS_SEGMENT_EXTRACTION
    assert get_max_input_chars("risk_summarization", provider="ollama") == OLLAMA_MAX_INPUT_CHARS_RISK_SUMMARIZATION
    assert get_max_input_chars("segment_extraction", provider="ollama") < CLOUD_MAX_INPUT_CHARS
    assert get_max_input_chars("risk_summarization", provider="ollama") < CLOUD_MAX_INPUT_CHARS


def test_risk_summarization_gets_a_smaller_ollama_budget_than_segment_extraction():
    """Risk summarization's own output can run much larger (multiple
    categories x several bullets each), competing with the input for the
    same context window -- it needs to leave more room than segment
    extraction, whose output is just a handful of segments."""
    assert (
        get_max_input_chars("risk_summarization", provider="ollama")
        < get_max_input_chars("segment_extraction", provider="ollama")
    )


def test_defaults_to_llm_provider_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert get_max_input_chars("segment_extraction") == OLLAMA_MAX_INPUT_CHARS_SEGMENT_EXTRACTION


def test_raises_on_unknown_task():
    try:
        get_max_input_chars("not_a_real_task", provider="ollama")
        assert False, "expected ValueError"
    except ValueError:
        pass
