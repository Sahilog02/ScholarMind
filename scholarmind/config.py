"""
LLM client factory. Swap providers with the LLM_PROVIDER env var so the same
agent code runs on the free Groq tier or on OpenAI without changes.
"""
import os


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'groq' or 'openai'.")
