#!/usr/bin/env python3
"""
Збирає зведення з evaluation/summary.json і показує метрики.

  # Після повного eval:
  python main.py --eval --output trajectories
  python scripts/show_eval_summary.py

  # Повний JSON (як раніше):
  python scripts/show_eval_summary.py --json
  python scripts/show_eval_summary.py trajectories/summary.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

W = 62


def _bar(label: str, value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:22s} [{bar}] {value * 100:5.1f}%"


def print_pretty(data: dict, source: str) -> None:
    print("\n" + "═" * W)
    print("  METRICS")
    print(f"  Файл: {source}")
    print("═" * W)

    print(f"\n  Модель:          {data.get('model', '—')}")
    print(f"  Час прогону:     {data.get('run_timestamp', '—')}")
    n = data.get("total_tasks", 0)
    tp = data.get("total_points", 0)
    mp = data.get("max_points", 0)
    pct = data.get("score_pct", 0)
    pc = data.get("pass_count", 0)
    pr = data.get("pass_rate", 0)
    if isinstance(pr, float) and pr <= 1.0:
        pr_pct = pr * 100
    else:
        pr_pct = float(pr or 0)

    print(f"\n  Задач:           {n}")
    print(f"  Бали:            {tp}/{mp}  ({pct}%)")
    print(f"  Pass (поріг):    {pc}/{n}  ({pr_pct:.0f}%)")
    print(f"  Latency (сер.):  {data.get('avg_latency_s', '—')}")
    print(f"  Tool calls (сер.): {data.get('avg_tool_calls', '—')}")

    if pr_pct is not None and n:
        _bar("Pass rate", pc / max(n, 1))

    by_cat = data.get("by_category") or {}
    if by_cat:
        print("\n  За категоріями:")
        for cat, s in sorted(by_cat.items()):
            tot = s.get("total", 0)
            ok = s.get("pass", 0)
            pts = s.get("points", 0)
            mx = s.get("max", 0)
            p = ok / tot * 100 if tot else 0
            print(f"    • {cat:18s}  {ok}/{tot} pass ({p:.0f}%)   {pts}/{mx} pt")

    tools = data.get("tool_usage") or {}
    if tools:
        print("\n  Використання MCP-тулів (усього викликів):")
        for name, cnt in sorted(tools.items(), key=lambda x: -x[1]):
            print(f"    • {name:26s}  {cnt}")

    print("\n" + "═" * W + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Показати метрики з trajectories/summary.json")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "trajectories" / "summary.json"),
        help="Шлях до summary.json (за замовч.: trajectories/summary.json)",
    )
    parser.add_argument("--json", action="store_true", help="Вивести повний JSON (як раніше)")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.is_file():
        print(f"Немає файлу: {path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Зібрати метрики:", file=sys.stderr)
        print("    python main.py --eval --output trajectories", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Потім знову:", file=sys.stderr)
        print("    python scripts/show_eval_summary.py", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_pretty(data, str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
