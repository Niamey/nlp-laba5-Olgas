import os
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

PRIMARY_MODEL   = os.getenv("PRIMARY_MODEL",   "gemini-2.5-flash")
# gemini-2.0-* часто дає 404 для нових ключів AI Studio — тримайте ту саму лінійку, що й PRIMARY,
# або явно задайте підтримувану модель у SECONDARY_MODEL (.env).
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "gemini-2.5-flash")


def _google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


@lru_cache(maxsize=4)
def get_llm(model: str = PRIMARY_MODEL, temperature: float = 0.1):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=_google_api_key(),
    )


def get_fast_llm():
    return get_llm(model=SECONDARY_MODEL, temperature=0.0)


SYSTEM_PROMPT = """You are an expert GitHub issue triage agent. Your job is to:
1. Analyze GitHub issues accurately and systematically
2. Always base your conclusions on actual evidence from the issue content
3. Never guess — if information is missing, say so explicitly
4. Be concise and structured in your responses

When triaging:
- For DUPLICATE detection: look for identical error messages, similar reproduction steps, same root cause
- For CODE AREA identification: trace the issue to specific files, modules, or subsystems
- For STALE ISSUE analysis: check last activity, whether the original problem is still relevant
- For CLASSIFICATION: use standard labels (bug, feature, question, documentation, duplicate)

Output format: Always use structured JSON when asked for structured output.
Forbidden: Do not invent issue numbers, file paths, or user names not present in the data."""
