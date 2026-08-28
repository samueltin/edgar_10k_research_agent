"""Tests for the provider-agnostic LLM factory. No real LLM/network calls
-- each branch's import is monkeypatched to confirm get_llm() dispatches
correctly and passes the right config through, without needing any of the
three provider packages actually installed.
"""
import sys
import types
import pytest


def _install_fake_module(monkeypatch, module_path, class_name, capture: dict):
    fake_module = types.ModuleType(module_path)

    class FakeClass:
        def __init__(self, **kwargs):
            capture.update(kwargs)

    setattr(fake_module, class_name, FakeClass)
    monkeypatch.setitem(sys.modules, module_path, fake_module)


def test_defaults_to_azure_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    captured = {}
    _install_fake_module(monkeypatch, "langchain_openai", "AzureChatOpenAI", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm()

    assert captured["azure_deployment"] == "gpt-4o"


def test_dispatches_to_anthropic_via_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    captured = {}
    _install_fake_module(monkeypatch, "langchain_anthropic", "ChatAnthropic", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm()

    assert captured["model"] == "claude-sonnet-4-5"


def test_dispatches_to_ollama_via_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)

    captured = {}
    _install_fake_module(monkeypatch, "langchain_ollama", "ChatOllama", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm()

    assert captured["base_url"] == "http://localhost:11434"
    assert captured["model"] == "llama3.1:8b"
    assert captured["num_ctx"] == 4096


def test_ollama_reads_custom_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.50:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")

    captured = {}
    _install_fake_module(monkeypatch, "langchain_ollama", "ChatOllama", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm()

    assert captured["base_url"] == "http://192.168.1.50:11434"
    assert captured["model"] == "qwen2.5:7b"
    assert captured["num_ctx"] == 16384


def test_explicit_provider_argument_overrides_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")

    captured = {}
    _install_fake_module(monkeypatch, "langchain_ollama", "ChatOllama", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm(provider="ollama")

    assert captured["model"] == "llama3.1:8b"


def test_raises_on_unknown_provider(monkeypatch):
    from edgar_research_agent.agent.llm_provider import get_llm

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm(provider="not_a_real_provider")


def test_temperature_is_passed_through(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    captured = {}
    _install_fake_module(monkeypatch, "langchain_ollama", "ChatOllama", captured)

    from edgar_research_agent.agent.llm_provider import get_llm
    get_llm(temperature=0.7)

    assert captured["temperature"] == 0.7
