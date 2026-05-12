"""Додаткові поля для JSON-траєкторії (звітність PDF / LangSmith-альтернатива)."""
from __future__ import annotations

from typing import Any


def _preview(content: Any, limit: int = 400) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        text = str(content)
    else:
        text = str(content)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def message_trace_from_state(state: dict, *, max_messages: int = 35) -> list[dict[str, Any]]:
    """
    Компактний журнал повідомлень з кінця графа (machine-readable).
    Gemini/OpenAI часто кладуть usage у response_metadata.
    """
    raw = state.get("messages") or []
    out: list[dict[str, Any]] = []
    tail = raw[-max_messages:] if len(raw) > max_messages else raw
    for m in tail:
        cls = type(m).__name__
        row: dict[str, Any] = {
            "type": cls,
            "content_preview": _preview(getattr(m, "content", None)),
        }
        rm = getattr(m, "response_metadata", None)
        if rm:
            row["response_metadata"] = rm
        tu = getattr(m, "usage_metadata", None)
        if tu:
            row["usage_metadata"] = tu
        out.append(row)
    return out


def aggregate_token_usage(message_trace: list[dict]) -> dict[str, Any]:
    """Сумує input/output токени з usage_metadata по повідомленнях (якщо є)."""
    inp = out = 0
    for row in message_trace:
        u = row.get("usage_metadata") or {}
        # LangChain Google: often input_tokens / output_tokens
        inp += int(u.get("input_tokens") or u.get("prompt_token_count") or 0)
        out += int(u.get("output_tokens") or u.get("candidates_token_count") or 0)
    return {"input_tokens_est": inp, "output_tokens_est": out, "total_tokens_est": inp + out}
