"""
Regex-витяг конкретних фактів зі звіту й перевірка проти ground-truth blob.
Швидко, без LLM. Слугує "second opinion" для grounding_validator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Patterns ──────────────────────────────────────────────────────
# Перевіряємо лише ті типи фактів, які реально галюцинуються в репортах.

_ISSUE_NUM_RE = re.compile(r"#(\d{2,6})\b")
_USERNAME_RE = re.compile(r"(?<![\w/])@([A-Za-z0-9][A-Za-z0-9\-]{0,38})\b")
_FILE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_/])"
    r"([a-zA-Z0-9_][a-zA-Z0-9_\-./]*?/[a-zA-Z0-9_\-./]*?"
    r"\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|md|json|yml|yaml|toml|ini|cfg|html|css|sh|ps1))"
)
_URL_RE = re.compile(r"https?://[^\s)\]>'\"`]+", re.IGNORECASE)
# Лише числа з контекстом «issue/PR» (інакше зловимо random цифри типу версій).
_ISSUE_KEYWORD_NUM_RE = re.compile(
    r"\b(?:issue|pr|pull[\s-]?request|ticket|item)\s*#?\s*(\d{2,6})\b",
    re.IGNORECASE,
)

# Ignore-list для usernames: системні / шаблонні згадки, не справжні юзери GitHub
_USERNAME_IGNORE = {
    "user", "users", "github", "bot", "ai", "llm", "agent",
    "owner", "maintainer", "author", "team", "anyone", "someone",
    "n/a", "none", "example", "username",
}


@dataclass
class FactCheckResult:
    issue_numbers:    list[str] = field(default_factory=list)
    usernames:        list[str] = field(default_factory=list)
    file_paths:       list[str] = field(default_factory=list)
    urls:             list[str] = field(default_factory=list)
    invented:         list[dict] = field(default_factory=list)
    total_facts:      int = 0
    grounded_facts:   int = 0
    facts_grounded_rate: float | None = None
    passed:           bool = True

    def to_dict(self) -> dict:
        return {
            "issue_numbers":       self.issue_numbers,
            "usernames":           self.usernames,
            "file_paths":          self.file_paths,
            "urls":                self.urls,
            "invented":            self.invented,
            "total_facts":         self.total_facts,
            "grounded_facts":      self.grounded_facts,
            "facts_grounded_rate": self.facts_grounded_rate,
            "passed":              self.passed,
        }


def _extract_unique(pattern: re.Pattern[str], text: str) -> list[str]:
    seen: list[str] = []
    found: set[str] = set()
    for m in pattern.finditer(text):
        val = m.group(1) if m.lastindex else m.group(0)
        key = val.lower()
        if key in found:
            continue
        found.add(key)
        seen.append(val)
    return seen


def extract_facts_from_report(report: str) -> dict[str, list[str]]:
    """
    Витягує перевірювані факти зі звіту.
    Повертає словник: {issue_numbers, usernames, file_paths, urls}.
    """
    if not report:
        return {"issue_numbers": [], "usernames": [], "file_paths": [], "urls": []}

    issue_nums_hash = _extract_unique(_ISSUE_NUM_RE, report)
    issue_nums_kw = _extract_unique(_ISSUE_KEYWORD_NUM_RE, report)
    issue_nums: list[str] = []
    seen: set[str] = set()
    for n in issue_nums_hash + issue_nums_kw:
        if n in seen:
            continue
        seen.add(n)
        issue_nums.append(n)

    raw_users = _extract_unique(_USERNAME_RE, report)
    usernames = [u for u in raw_users if u.lower() not in _USERNAME_IGNORE]

    file_paths = _extract_unique(_FILE_PATH_RE, report)
    urls = _extract_unique(_URL_RE, report)

    return {
        "issue_numbers": issue_nums,
        "usernames":     usernames,
        "file_paths":    file_paths,
        "urls":          urls,
    }


def _present_in_blob(needle: str, blob_lower: str, *, mode: str) -> bool:
    """Чи зустрічається factu blob (lowercased)."""
    if not needle:
        return True
    n = needle.lower()
    if mode == "issue_num":
        # допускаємо `#1234`, `issue 1234`, `pr 1234`, just `1234` ізольоване
        if f"#{n}" in blob_lower:
            return True
        if re.search(rf"\b{re.escape(n)}\b", blob_lower):
            return True
        return False
    if mode == "user":
        if f"@{n}" in blob_lower:
            return True
        if re.search(rf"\b{re.escape(n)}\b", blob_lower):
            return True
        return False
    if mode == "path":
        if n in blob_lower:
            return True
        # допускаємо basename (агент часто скорочує)
        base = n.rsplit("/", 1)[-1]
        if base and base in blob_lower:
            return True
        return False
    if mode == "url":
        return n in blob_lower
    return n in blob_lower


def check_facts_against_blob(
    report: str,
    ground_truth_blob: str,
    *,
    fail_threshold: float = 0.8,
) -> FactCheckResult:
    """
    Перевіряє кожен витягнутий факт у звіті проти ground truth blob.

    Args:
        report:             текст triage report
        ground_truth_blob:  склеєний blob з tool_results + findings + issue body
        fail_threshold:     pass якщо facts_grounded_rate >= threshold (default 0.8)

    Returns:
        FactCheckResult зі списком знайденого, invented[], total/grounded/rate, passed.
    """
    facts = extract_facts_from_report(report)
    blob_lower = (ground_truth_blob or "").lower()

    invented: list[dict] = []
    total = 0
    grounded = 0

    categories = [
        ("issue_num", "issue_numbers", facts["issue_numbers"]),
        ("user",      "usernames",     facts["usernames"]),
        ("path",      "file_paths",    facts["file_paths"]),
        ("url",       "urls",          facts["urls"]),
    ]

    for mode, label, items in categories:
        for v in items:
            total += 1
            if _present_in_blob(v, blob_lower, mode=mode):
                grounded += 1
            else:
                invented.append({"kind": label, "value": v})

    rate: float | None
    if total == 0:
        rate = None
        passed = True
    else:
        rate = round(grounded / total, 3)
        passed = rate >= fail_threshold

    return FactCheckResult(
        issue_numbers=facts["issue_numbers"],
        usernames=facts["usernames"],
        file_paths=facts["file_paths"],
        urls=facts["urls"],
        invented=invented,
        total_facts=total,
        grounded_facts=grounded,
        facts_grounded_rate=rate,
        passed=passed,
    )


def format_fact_check_warning(fc: FactCheckResult, max_items: int = 5) -> str:
    """Короткий human-readable рядок для логів."""
    if fc.total_facts == 0:
        return "fact_check: no checkable facts in report"
    pct = (fc.facts_grounded_rate or 0) * 100
    status = "PASS" if fc.passed else "FAIL"
    base = f"fact_check {status}: {fc.grounded_facts}/{fc.total_facts} grounded ({pct:.0f}%)"
    if fc.invented:
        items = ", ".join(f"{x['kind']}={x['value']!r}" for x in fc.invented[:max_items])
        more = f" (+{len(fc.invented) - max_items} more)" if len(fc.invented) > max_items else ""
        return f"{base}; INVENTED: {items}{more}"
    return base
