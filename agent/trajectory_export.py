"""Додаткові поля для JSON-траєкторії (звітність PDF / LangSmith-альтернатива)."""
from __future__ import annotations

import numbers
from typing import Any


def _preview(content: Any, limit: int = 400) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        text = str(content)
    else:
        text = str(content)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def message_trace_from_state(state: dict, *, max_messages: int | None = None) -> list[dict[str, Any]]:
    """
    Компактний журнал повідомлень з кінця графа (machine-readable).
    Gemini/OpenAI часто кладуть usage у response_metadata.
    max_messages=None — усі повідомлення (потрібно для суми токенів у довгих графах).
    """
    raw = state.get("messages") or []
    out: list[dict[str, Any]] = []
    if max_messages is None or max_messages <= 0:
        tail = raw
    else:
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


def _as_int(v: Any) -> int:
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, float):
        return int(v)
    return 0


def _tokens_from_usage_dict(u: dict) -> tuple[int, int]:
    if not u:
        return 0, 0
    inp = _as_int(
        u.get("input_tokens")
        or u.get("prompt_token_count")
        or u.get("input_token_count")
        or u.get("prompt_tokens")
    )
    out = _as_int(
        u.get("output_tokens")
        or u.get("candidates_token_count")
        or u.get("output_token_count")
        or u.get("completion_tokens")
    )
    return inp, out


def _walk_for_usage(obj: Any, depth: int = 0) -> tuple[int, int]:
    """Рекурсивно шукає поля токенів у response_metadata (різні провайдери)."""
    if depth > 12 or obj is None:
        return 0, 0
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if "prompt_token_count" in obj or "candidates_token_count" in obj:
            return _tokens_from_usage_dict(obj)
        if "usage_metadata" in obj and isinstance(obj["usage_metadata"], dict):
            return _tokens_from_usage_dict(obj["usage_metadata"])
        if "usage" in obj and isinstance(obj["usage"], dict):
            return _tokens_from_usage_dict(obj["usage"])
        acc_i = acc_o = 0
        for v in obj.values():
            di, do = _walk_for_usage(v, depth + 1)
            acc_i += di
            acc_o += do
        return acc_i, acc_o
    if isinstance(obj, (list, tuple)):
        ai = ao = 0
        for x in obj:
            di, do = _walk_for_usage(x, depth + 1)
            ai += di
            ao += do
        return ai, ao
    return 0, 0


def aggregate_token_usage(message_trace: list[dict]) -> dict[str, Any]:
    """Сумує input/output токени з usage_metadata та response_metadata по повідомленнях."""
    inp = out = 0
    for row in message_trace:
        di = do = 0
        u = row.get("usage_metadata") or {}
        if isinstance(u, dict):
            di, do = _tokens_from_usage_dict(u)
        if di or do:
            inp += di
            out += do
            continue
        rm = row.get("response_metadata")
        if isinstance(rm, dict):
            ri, ro = _walk_for_usage(rm)
            inp += ri
            out += ro
    return {"input_tokens_est": inp, "output_tokens_est": out, "total_tokens_est": inp + out}
