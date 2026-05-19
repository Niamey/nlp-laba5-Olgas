#!/usr/bin/env python3
"""Показати метрики з уже збереженого JSON траєкторії (без повторного тріажу).

  python scripts/show_trajectory_metrics.py trajectories/4a43d9ac-42a8-4214-baba-c9fd4b9c3e1b.json
  python scripts/show_trajectory_metrics.py trajectories/4a43d9ac-42a8-4214-baba-c9fd4b9c3e1b.json --json
  python scripts/show_trajectory_metrics.py trajectories/T002.json --extended
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import print_trajectory_metrics_explained, _strip_message_trace_for_print  # noqa: E402
from agent.extended_metrics import compute_extended_metrics, print_extended_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Вивід метрик з trajectory JSON.")
    ap.add_argument("trajectory_json", type=Path, help="Шлях до .json (trajectories/…)")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Після пояснення друкувати компактний JSON (без message_trace)",
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help="Розширені метрики (обчислити з файлу, якщо extended_metrics ще немає)",
    )
    ap.add_argument(
        "--task-id",
        help="Критерії з tasks.json для rubric_autocheck (напр. T002)",
    )
    args = ap.parse_args()
    p = args.trajectory_json
    if not p.is_file():
        print(f"Файл не знайдено: {p}", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    if not args.extended:
        print_trajectory_metrics_explained(data)

    if args.extended:
        traj = data.get("trajectory") if isinstance(data.get("trajectory"), dict) else data
        issue_url = data.get("issue_url") or traj.get("issue_url") or ""
        em = data.get("extended_metrics")
        if not isinstance(em, dict):
            task = None
            tid = (args.task_id or data.get("task_id") or "").strip()
            if tid:
                tasks_path = ROOT / "evaluation" / "tasks.json"
                if tasks_path.is_file():
                    for t in json.loads(tasks_path.read_text(encoding="utf-8")):
                        if t.get("task_id") == tid:
                            task = t
                            break
            em = compute_extended_metrics(
                report=data.get("full_report") or data.get("report") or "",
                state={},
                trajectory=traj,
                task=task,
            )
        print_extended_metrics(em, issue_url=issue_url)

    if args.json:
        small = _strip_message_trace_for_print(data)
        print(json.dumps(small, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
