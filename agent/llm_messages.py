"""Допоміжні функції для збереження usage_metadata у AIMessage (звітність токенів у eval)."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def aimessage_from_llm(content: str, llm_response: BaseMessage) -> AIMessage:
    """
    Створює AIMessage для add_messages, копіюючи usage/response metadata з реальної відповіді LLM.
    Без цього aggregate_token_usage бачить лише «ручні» AIMessage без токенів.
    """
    extra: dict[str, Any] = {}
    um = getattr(llm_response, "usage_metadata", None)
    if um is not None:
        if hasattr(um, "model_dump"):
            extra["usage_metadata"] = um.model_dump()
        elif isinstance(um, dict):
            extra["usage_metadata"] = dict(um)
        else:
            try:
                extra["usage_metadata"] = dict(um)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
    rm = getattr(llm_response, "response_metadata", None)
    if isinstance(rm, dict) and rm:
        extra["response_metadata"] = dict(rm)
    return AIMessage(content=content, **extra)
