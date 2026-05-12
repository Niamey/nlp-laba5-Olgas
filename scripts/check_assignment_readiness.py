#!/usr/bin/env python3
"""
Перевірка базових вимог до набору задач Track C перед eval/захистом.
Запуск з кореня репозиторію: python scripts/check_assignment_readiness.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "evaluation" / "tasks.json"


def main() -> int:
    if not TASKS_PATH.is_file():
        print(f"ERROR: немає {TASKS_PATH}", file=sys.stderr)
        return 1
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    n = len(tasks)
    cats = Counter(t.get("category", "?") for t in tasks)

    repos: set[tuple[str, str]] = set()
    for t in tasks:
        url = (t.get("input") or {}).get("issue_url") or ""
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues/", url)
        if m:
            repos.add((m.group(1), m.group(2)))

    print(f"tasks.json: {n} задач")
    print("Категорії:")
    for k, v in sorted(cats.items()):
        print(f"  {k}: {v}")

    warn: list[str] = []
    if n < 30:
        warn.append(f"Занадто мало задач: {n} (зазвичай очікують >= 30).")
    if cats.get("adversarial", 0) < 3:
        warn.append("Adversarial < 3 — додай хитрі edge cases.")
    if cats.get("should_refuse", 0) < 1:
        warn.append("Немає should_refuse — додай задачі «відмова / out of scope».")
    if cats.get("ambiguous", 0) < 2:
        warn.append("Мало ambiguous — для PDF корисні неоднозначні кейси.")

    print("\nУнікальні org/repo у URL:")
    for org, repo in sorted(repos):
        print(f"  {org}/{repo}")

    print("\nДалі:")
    print("  python main.py --list-mcp")
    print("  python main.py --eval --output trajectories/")
    print("  python scripts/show_eval_summary.py")

    if warn:
        print("\nЗауваги:")
        for w in warn:
            print(f"  - {w}")
        return 2
    print("\nOK: базові пороги задоволені.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
