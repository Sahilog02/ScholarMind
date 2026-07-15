"""
LLM client factory. Swap providers with the LLM_PROVIDER env var so the same
agent code runs on free OpenRouter models, Gemini, Groq, or OpenAI without
changes.

Default provider is OpenRouter using "openrouter/free" -- OpenRouter's free
model router, which automatically selects an available free, open-source
model (and supports structured outputs / tool calling, which this project
relies on via with_structured_output()). Get a free key at
https://openrouter.ai/keys -- no credit card required.

Set LLM_PROVIDER=gemini to use Google's Gemini API instead (also free with
rate limits) -- get a key at https://aistudio.google.com/apikey.
"""
import os


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Get a free key at "
                "https://openrouter.ai/keys and put it in your .env file."
            )

        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
            temperature=temperature,
            default_headers={
                # Optional but recommended by OpenRouter for attribution/rate-limit purposes.
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "ScholarMind"),
            },
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in your .env file."
            )

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=temperature,
        )

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

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
