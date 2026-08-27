"""Provider-agnostic LLM factory.

Depending on other code on this abstraction (rather than importing
AzureChatOpenAI directly in graph nodes) means swapping providers is a
config change, not a rewrite. Both classes implement LangChain's
BaseChatModel interface, so `.with_structured_output(...)` works identically
either way.
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
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
