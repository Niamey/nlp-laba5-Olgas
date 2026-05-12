"""
Code references for report (Track A).

This file mirrors the *actual* logic used in the Kaggle notebook:
- Track choice (A-only) and task format
- Agent role/system prompt (what the assistant is)
- What counts as success/failure in evaluation:
  - grounded identifiers (mentions ⊆ tool evidence)
  - tool-selection vs rubric.must_use_tool_classes

It exists so you can cite stable line numbers in your report.
Canonical implementation lives in `nlp_assignment5_trackA_kaggle.ipynb`.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict


class Task(TypedDict, total=False):
    id: str
    track: str
    adversarial: bool
    prompt: str
    rubric: dict[str, Any]


def format_task_prompt(task_dict: Task) -> str:
    # Track A only: the user query is task_dict["prompt"]
    return str(task_dict.get("prompt", ""))


def variant_system_prompt(*, ablation_variant: str = "A") -> str:
    base_common = """You are a Track-A scientific literature assistant.
Ground every factual claim using MCP tools only. Prefer arXiv + Semantic Scholar/OpenAlex graphs.
Explicitly cite arxiv identifiers like `arxiv:YYMM.NNNNN`. Never invent DOIs or arXiv ids.
Respect polite rate limits; batch queries."""
    if ablation_variant == "B":
        return base_common + " Tool discipline: NEVER call redundant tools — each batch must widen evidence."
    return base_common + " Keep user-facing prose compact."


# The evaluation checks for arXiv-like ids in the final assistant text.
ARXIV_ID_REGEX = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def grounding_stats(final_assistant_text: str, tool_evidence_blob: str) -> tuple[set[str], set[str]]:
    """
    Success condition for "grounded IDs":
      mentions = arXiv-like ids in final assistant text
      grounded = subset of mentions that appear in ToolMessage evidence blob
    Failure (ungrounded) if (mentions - grounded) is non-empty.
    """
    mentions = set(ARXIV_ID_REGEX.findall(final_assistant_text or ""))
    grounded = set(x for x in mentions if x in (tool_evidence_blob or ""))
    return mentions, grounded


def tool_selection_ok(*, used_tool_classes: set[str], must_use_tool_classes: list[str]) -> bool | None:
    """
    Success condition for tool selection (rubric):
      For tasks with must_use_tool_classes: all required classes must appear in used_tool_classes.
    If rubric has no requirements: return None (not scored).
    """
    req = [str(x) for x in (must_use_tool_classes or [])]
    if not req:
        return None
    return all(x in used_tool_classes for x in req)


def extract_must_use_tool_classes(task: Task) -> list[str]:
    rubric = task.get("rubric") or {}
    classes = rubric.get("must_use_tool_classes") or []
    return [str(x) for x in classes]


def load_tasks_embedded(json_text: str) -> list[Task]:
    tasks = json.loads(json_text)
    # Track A-only invariant (the notebook asserts this)
    assert all(t.get("track") == "A" for t in tasks)
    return tasks

