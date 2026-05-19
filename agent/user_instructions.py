"""Блок додаткових інструкцій користувача для промптів графа."""
from __future__ import annotations

from pathlib import Path
from typing import Any

VALID_TASK_TYPES = frozenset({"duplicate", "code_area", "stale", "classify"})


def normalize_task_hint(hint: str | None) -> str | None:
    """Перше слово з hint, якщо це один із режимів тріажу."""
    if not hint or not str(hint).strip():
        return None
    low = str(hint).strip().lower()
    first = low.split()[0] if low else ""
    if low in VALID_TASK_TYPES:
        return low
    if first in VALID_TASK_TYPES:
        return first
    return None


def load_user_prompt_from_cli(cli_prompt: str | None, prompt_file: str | None) -> str | None:
    """Текст з --prompt або --prompt-file (файл має пріоритет над --prompt)."""
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8").strip()
        if text:
            return text
    if cli_prompt and str(cli_prompt).strip():
        return str(cli_prompt).strip()
    return None


def effective_user_prompt_for_task(task: dict, default_prompt: str | None) -> str | None:
    """Промпт задачі з tasks.json input.user_prompt, інакше загальний з CLI."""
    inp = task.get("input") or {}
    per_task = inp.get("user_prompt")
    if per_task and str(per_task).strip():
        return str(per_task).strip()
    return default_prompt


def user_prompt_text(state: dict[str, Any]) -> str:
    return (state.get("user_prompt") or "").strip()


def user_instructions_block(state: dict[str, Any], *, prefix: str = "") -> str:
    """
    Текст для додавання в HumanMessage після основного промпту.
    Порожній рядок, якщо user_prompt не задано.
    """
    text = user_prompt_text(state)
    if not text:
        return ""
    head = prefix.strip()
    if head:
        return f"\n\n{head}\n{text}"
    return f"\n\nUser instructions (follow these in addition to the task above):\n{text}"
