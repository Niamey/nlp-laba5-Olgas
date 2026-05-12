"""
evaluation/runner.py — запускає всі 30 задач і зберігає результати.
Логує: tool calls, latency, grounding, success/fail per task.
"""
from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("eval")


TASKS_FILE = Path(__file__).parent / "tasks.json"


async def score_result(task: dict, result: dict) -> dict:
    """
    Скорить результат за рубрикою задачі.
    Повертає dict з: points, max_points, pass/fail, reasons.
    """
    criteria   = task.get("success_criteria", {})
    report     = result.get("report", "").lower()
    trajectory = result.get("trajectory", {})
    state      = result.get("state", {})

    reasons  = []
    demerits = []
    score    = task["scoring"]["max_points"]

    # Перевіряємо заборонені поведінки
    for fb in task.get("forbidden_behaviors", []):
        if fb == "inventing issue numbers not found by search":
            # Перевіряємо чи є в звіті issue numbers яких немає в tool_results
            pass  # складна перевірка, пропускаємо в базовому runner
        elif fb == "crashing":
            if not result.get("report"):
                demerits.append("Agent crashed (no report)")
                score = 0

    # Перевіряємо обов'язкові елементи
    if criteria.get("must_not_crash") and not result.get("report"):
        demerits.append("FAIL: Agent produced no report")
        score = 0

    if criteria.get("must_report_not_found"):
        if "not found" in report or "404" in report or "does not exist" in report:
            reasons.append("PASS: Correctly reported not found")
        else:
            demerits.append("FAIL: Did not report not found")
            score = max(0, score - 2)

    if criteria.get("grounding_required"):
        if state.get("grounding_passed"):
            reasons.append("PASS: Grounding check passed")
        else:
            demerits.append("WARN: Grounding check failed")
            score = max(0, score - 1)

    if criteria.get("must_search"):
        tools_called = [r.get("tool") for r in trajectory.get("tool_results", [])]
        if "search_similar_issues" in tools_called:
            reasons.append("PASS: Searched for similar issues")
        else:
            demerits.append("FAIL: Did not search for similar issues")
            score = max(0, score - 2)

    if criteria.get("must_identify_module"):
        module_keywords = ["module", "component", "file", "directory", "router", "middleware", "handler"]
        if any(kw in report for kw in module_keywords):
            reasons.append("PASS: Identified a module/component")
        else:
            demerits.append("WARN: No clear module identified")
            score = max(0, score - 1)

    if criteria.get("must_recommend_action"):
        action_keywords = ["recommend", "suggest", "close", "ping", "action", "next step"]
        if any(kw in report for kw in action_keywords):
            reasons.append("PASS: Included recommendation")
        else:
            demerits.append("FAIL: No actionable recommendation")
            score = max(0, score - 1)

    if criteria.get("type_correct"):
        correct_types = criteria["type_correct"]
        if any(t in report for t in correct_types):
            reasons.append(f"PASS: Correct type detected ({correct_types})")
        else:
            demerits.append(f"FAIL: Expected type {correct_types} not in report")
            score = max(0, score - 2)

    if criteria.get("must_not_hallucinate_issue_content"):
        # Якщо issue not found, але в звіті є детальний опис — підозріло
        if "not found" in report and len(report) > 500:
            demerits.append("WARN: Possible hallucination despite not-found")
            score = max(0, score - 1)

    final_score = max(0, min(score, task["scoring"]["max_points"]))

    return {
        "task_id":    task["task_id"],
        "score":      final_score,
        "max_points": task["scoring"]["max_points"],
        "passed":     final_score >= task["scoring"]["max_points"] * 0.66,
        "reasons":    reasons,
        "demerits":   demerits,
    }


async def run_evaluation(
    output_dir:   str = "trajectories",
    tasks_filter: Optional[list] = None,
    model:        Optional[str]  = None,
    no_hil:       bool = True,
):
    """
    Запускає всі задачі з tasks.json і зберігає результати.

    Args:
        output_dir:   директорія для trajectory JSON файлів
        tasks_filter: список task_id для запуску (None = всі)
        model:        override PRIMARY_MODEL env var
        no_hil:       True = вимкнути HIL для batch evaluation
    """
    import os
    if model:
        os.environ["PRIMARY_MODEL"] = model

    # Пізній імпорт щоб уникнути circular imports
    from main import run_triage

    tasks_file = TASKS_FILE
    with open(tasks_file) as f:
        all_tasks = json.load(f)

    if tasks_filter:
        tasks = [t for t in all_tasks if t["task_id"] in tasks_filter]
    else:
        tasks = all_tasks

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_ts  = datetime.now().isoformat()
    results = []

    logger.info("Starting evaluation: %d tasks", len(tasks))
    print(f"\n{'═'*60}")
    print(f"  EVALUATION RUN — {run_ts}")
    print(f"  Tasks: {len(tasks)} | Model: {os.getenv('PRIMARY_MODEL', 'default')}")
    print(f"{'═'*60}\n")

    for i, task in enumerate(tasks, 1):
        task_id  = task["task_id"]
        category = task.get("category", "?")
        print(f"[{i:02d}/{len(tasks)}] {task_id} ({category}) — {task['description'][:50]}...")

        start = time.time()
        try:
            result = await run_triage(
                issue_url=task["input"]["issue_url"],
                task_hint=task["input"].get("hint"),
                human_in_loop=not no_hil,
            )
            elapsed = round(time.time() - start, 2)

            score_result_data = await score_result(task, result)

            entry = {
                "task_id":    task_id,
                "category":   category,
                "type":       task.get("type"),
                "score":      score_result_data["score"],
                "max_points": score_result_data["max_points"],
                "passed":     score_result_data["passed"],
                "reasons":    score_result_data["reasons"],
                "demerits":   score_result_data["demerits"],
                "trajectory": result.get("trajectory", {}),
                "report_snippet": result.get("report", "")[:300],
                "elapsed_s":  elapsed,
                "error":      None,
            }

            # Зберігаємо повний trajectory
            traj_path = out / f"{task_id}.json"
            with open(traj_path, "w") as f:
                json.dump({**entry, "full_report": result.get("report")}, f, indent=2, default=str)

            status = "PASS" if entry["passed"] else "FAIL"
            print(f"       {status} {entry['score']}/{entry['max_points']}pt — {elapsed}s")

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error("Task %s crashed: %s", task_id, e)
            entry = {
                "task_id":    task_id,
                "category":   category,
                "score":      0,
                "max_points": task["scoring"]["max_points"],
                "passed":     False,
                "error":      str(e),
                "elapsed_s":  elapsed,
            }
            print(f"       ERROR — {e}")

        results.append(entry)

    # ── Aggregate metrics ─────────────────────────────────────────
    total_points  = sum(r.get("score", 0) for r in results)
    max_points    = sum(r.get("max_points", 0) for r in results)
    pass_count    = sum(1 for r in results if r.get("passed"))
    by_category: dict = {}

    for r in results:
        cat = r.get("category", "other")
        if cat not in by_category:
            by_category[cat] = {"pass": 0, "total": 0, "points": 0, "max": 0}
        by_category[cat]["total"]  += 1
        by_category[cat]["points"] += r.get("score", 0)
        by_category[cat]["max"]    += r.get("max_points", 0)
        if r.get("passed"):
            by_category[cat]["pass"] += 1

    # Tool usage stats
    all_tools = []
    for r in results:
        traj = r.get("trajectory", {})
        for tr in traj.get("tool_results", []):
            all_tools.append(tr.get("tool"))

    tool_counts = {}
    for t in all_tools:
        tool_counts[t] = tool_counts.get(t, 0) + 1

    avg_latency = sum(r.get("elapsed_s", 0) for r in results) / max(len(results), 1)
    avg_tools   = sum(
        r.get("trajectory", {}).get("tool_calls", 0) for r in results
    ) / max(len(results), 1)

    summary = {
        "run_timestamp":  run_ts,
        "model":          model or os.getenv("PRIMARY_MODEL", "default"),
        "total_tasks":    len(tasks),
        "pass_count":     pass_count,
        "pass_rate":      round(pass_count / max(len(tasks), 1), 3),
        "total_points":   total_points,
        "max_points":     max_points,
        "score_pct":      round(total_points / max(max_points, 1) * 100, 1),
        "avg_latency_s":  round(avg_latency, 2),
        "avg_tool_calls": round(avg_tools, 2),
        "tool_usage":     tool_counts,
        "by_category":    by_category,
        "results":        results,
    }

    # Зберігаємо summary
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Печатаємо результати
    print(f"\n{'═'*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  Score:   {total_points}/{max_points} ({summary['score_pct']}%)")
    print(f"  Pass:    {pass_count}/{len(tasks)} ({summary['pass_rate']*100:.0f}%)")
    print(f"  Latency: {avg_latency:.1f}s avg")
    print(f"  Tools:   {avg_tools:.1f} avg per task")
    print(f"\n  By category:")
    for cat, s in by_category.items():
        pct = s["pass"] / max(s["total"], 1) * 100
        print(f"    {cat:20s} {s['pass']}/{s['total']} ({pct:.0f}%) — {s['points']}/{s['max']}pt")
    print(f"\n  Tool usage: {tool_counts}")
    print(f"\n  Full results: {summary_path}")
    print(f"{'═'*60}\n")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  nargs="+", help="Specific task IDs to run")
    parser.add_argument("--model",  help="Override model")
    parser.add_argument("--output", default="trajectories")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        output_dir=args.output,
        tasks_filter=args.tasks,
        model=args.model,
    ))
