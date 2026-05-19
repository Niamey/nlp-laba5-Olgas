"""Нормалізація мітки типу issue для classify worker і eval rubric."""
from __future__ import annotations


def normalize_issue_type(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t:
        return ""
    if t in ("documentation", "document", "doc", "docs"):
        return "documentation"
    if t in ("feature", "feat", "enhancement", "enh"):
        return "feature"
    if t in ("bug", "defect"):
        return "bug"
    if t in ("question", "support"):
        return "question"
    if t == "duplicate":
        return "duplicate"
    if "document" in t:
        return "documentation"
    if "duplicate" in t:
        return "duplicate"
    if "question" in t:
        return "question"
    if "feature" in t:
        return "feature"
    if "bug" in t:
        return "bug"
    return t
