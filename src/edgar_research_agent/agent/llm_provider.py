"""Provider-agnostic LLM factory.

Depending on other code on this abstraction (rather than importing
AzureChatOpenAI directly in graph nodes) means swapping providers is a
config change, not a rewrite. All three classes implement LangChain's
BaseChatModel interface, so `.with_structured_output(...)` works
identically across all of them.

"ollama" runs a local model on your own hardware (e.g. an Ollama server
on a separate Linux PC on your LAN), eliminating Azure OpenAI API cost
entirely. Real trade-offs, not a free swap -- see llm_provider's ollama
branch and docs/architecture.md for what to actually expect from a local
7-8B model on this pipeline's specific tasks.
"""
import os


def get_llm(provider: str | None = None, temperature: float = 0):
    provider = provider or os.environ.get("LLM_PROVIDER", "azure_openai")

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        # num_ctx matters more here than it might elsewhere: Ollama models
        # often default to a small context window (sometimes 2048 tokens)
        # regardless of what the underlying model architecture actually
        # supports, unless told otherwise. CONFIRMED ON REAL HARDWARE: a
        # real llama3.1:8b run on an 8GB GPU reported a 4096-token context
        # window -- see context_limits.py for how the pipeline's prompts
        # are now sized to actually fit within that, rather than assuming
        # a larger window and hoping for the best.
        return ChatOllama(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=temperature,
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
