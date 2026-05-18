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


def print_metrics_help() -> None:
    """Друкує пояснення метрик eval у консоль (виклик з CLI: --metrics-help)."""
    print(
        """
======================================================================
  Метрики evaluation (файл evaluation/runner.py)
======================================================================

1) За кожною задачею (після run_triage):
   - score / max_points — автоматична евристика з success_criteria у tasks.json.
   - passed — True, якщо score >= 66% від max_points (див. score_result).

   Що перевіряє score_result, якщо відповідний ключ є в success_criteria:
   - must_not_crash — є непорожній report.
   - must_report_not_found — у звіті є not found / 404 / does not exist.
   - grounding_required — grounding_passed == True (спочатку state графа,
     якщо None — береться trajectory.grounding_passed).
   - must_search — у trajectory.tool_results був виклик search_similar_issues.
   - must_identify_module — у звіті є слова module, component, file, ...
   - must_recommend_action — recommend, suggest, close, ping, action, next step.
   - type_correct — у тексті звіту згадується один із очікуваних типів.
   - must_not_hallucinate_issue_content — not-found, але дуже довгий звіт.

   reasons / demerits зберігаються в trajectories/<TASK>.json і в
   trajectories/summary.json у масиві results.

2) Підсумок після всіх задач (консоль і trajectories/summary.json):
   - total_points, max_points, score_pct — сума балів.
   - pass_count, pass_rate — скільки задач з passed=True.
   - avg_latency_s, avg_tool_calls — середні значення.
   - grounding — лічильники true/false/unknown і pass_rate_decided.
   - token_usage_est_total — сума оцінок токенів з trajectory.
   - by_category — успіх і бали по category.
   - tool_usage — частота викликів кожної тули.

3) Детальний друк у консоль під час eval:
     python main.py --eval -v
   або: python -m evaluation.runner -v
   (після кожної задачі: reasons, demerits, grounding_passed).
""".strip()
    )


def print_issue_type_examples(tasks_path: Path | None = None) -> None:
    """Один приклад URL на кожен type з tasks.json (для швидкого ручного тесту)."""
    path = tasks_path or TASKS_FILE
    with open(path, encoding="utf-8") as f:
        tasks: list = json.load(f)
    order = ["classify", "duplicate", "code_area", "stale", "any", "out_of_scope"]
    picked: dict[str, dict] = {}
    for t in tasks:
        typ = t.get("type") or "?"
        if typ not in picked:
            picked[typ] = t
    print(
        "\nПриклади issue з evaluation/tasks.json (по одному на тип; "
        "hint — з поля input.hint):\n"
    )
    print(f"{'type':<14} {'task':<6} {'hint':<12} issue_url")
    print("-" * 72)
    for typ in order:
        t = picked.get(typ)
        if not t:
            continue
        inp = t.get("input") or {}
        hint = inp.get("hint")
        hint_s = (hint if hint is not None else "—")[:12]
        print(f"{typ:<14} {t.get('task_id', '?'):<6} {hint_s:<12} {inp.get('issue_url', '')}")
    extra = [typ for typ in sorted(picked) if typ not in order]
    for typ in extra:
        t = picked[typ]
        inp = t.get("input") or {}
        hint = inp.get("hint")
        hint_s = (hint if hint is not None else "—")[:12]
        print(f"{typ:<14} {t.get('task_id', '?'):<6} {hint_s:<12} {inp.get('issue_url', '')}")
    print(
        "\nЗапуск одного кейсу (PowerShell):\n"
        '  python main.py --url "<URL>" --hint duplicate --no-hil\n'
        "Повний eval з деталями:\n"
        "  python main.py --eval -v --output trajectories\n"
    )


def _grounding_passed_effective(result: dict) -> bool | None:
    """True/False з state графа; якщо None — з trajectory (для скорингу)."""
    state = result.get("state") or {}
    gp = state.get("grounding_passed")
    if gp is not None:
        return bool(gp)
    traj = result.get("trajectory") or {}
    gp2 = traj.get("grounding_passed")
    if gp2 is None:
        return None
    return bool(gp2)


def aggregate_triage_metrics(results: list[dict], task_by_id: dict[str, dict]) -> dict:
    """
    Додаткові метрики оцінки тріажу: розподіл балів, тип сценарію, grounding за рубрикою,
    сигнали rubric (reasons/demerits).
    """
    n = len(results)
    if not n:
        return {}

    hist: dict[str, int] = {}
    by_type: dict[str, dict] = {}
    perfect = 0
    err = 0
    g_req_total = 0
    g_req_ok = 0
    inconsistent = 0
    total_reasons = 0
    total_demerits = 0

    for r in results:
        if r.get("error"):
            err += 1
            hist["0"] = hist.get("0", 0) + 1
            continue
        sc = int(r.get("score") or 0)
        mx = int(r.get("max_points") or 0)
        key = str(sc)
        hist[key] = hist.get(key, 0) + 1
        if mx and sc == mx:
            perfect += 1

        tid = r.get("task_id") or ""
        task = task_by_id.get(tid, {})
        typ = (task.get("type") or r.get("type") or "?") or "?"
        if typ not in by_type:
            by_type[typ] = {"pass": 0, "total": 0, "points": 0, "max": 0}
        by_type[typ]["total"] += 1
        by_type[typ]["points"] += sc
        by_type[typ]["max"] += mx
        if r.get("passed"):
            by_type[typ]["pass"] += 1

        crit = (task.get("success_criteria") or {})
        if crit.get("grounding_required"):
            g_req_total += 1
            gp = _grounding_passed_effective({"state": {}, "trajectory": r.get("trajectory", {})})
            if gp is True:
                g_req_ok += 1
            if r.get("passed") and gp is not True:
                inconsistent += 1

        total_reasons += len(r.get("reasons") or [])
        total_demerits += len(r.get("demerits") or [])

    ok_n = n - err
    return {
        "non_crash_tasks":         ok_n,
        "perfect_tasks":           perfect,
        "perfect_rate":            round(perfect / max(ok_n, 1), 3),
        "score_histogram":         hist,
        "by_task_type":            by_type,
        "grounding_when_required": {
            "tasks":      g_req_total,
            "grounded":   g_req_ok,
            "pass_rate":  round(g_req_ok / max(g_req_total, 1), 3) if g_req_total else None,
        },
        "passed_but_grounding_failed": inconsistent,
        "rubric_signals": {
            "avg_reasons_per_task":   round(total_reasons / max(n, 1), 2),
            "avg_demerits_per_task":  round(total_demerits / max(n, 1), 2),
            "total_pass_reasons":     total_reasons,
            "total_demerits":         total_demerits,
        },
        "crashed_tasks": err,
    }


async def score_result(task: dict, result: dict) -> dict:
    """
    Скорить результат за рубрикою задачі.
    Повертає dict з: points, max_points, pass/fail, reasons.
    """
    criteria   = task.get("success_criteria", {})
    report     = result.get("report", "").lower()
    trajectory = result.get("trajectory", {}) or {}
    state      = result.get("state", {}) or {}
    grounding_ok = _grounding_passed_effective(result)

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
        if grounding_ok is True:
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
    verbose:      bool = False,
):
    """
    Запускає всі задачі з tasks.json і зберігає результати.

    Args:
        output_dir:   директорія для trajectory JSON файлів
        tasks_filter: список task_id для запуску (None = всі)
        model:        override PRIMARY_MODEL env var
        no_hil:       True = вимкнути HIL для batch evaluation
        verbose:      друкувати reasons/demerits і grounding після кожної задачі
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
    if verbose:
        print(
            "  [verbose] Після кожної задачі: reasons (+), demerits (-), "
            "grounding_passed з trajectory.\n"
            "  Повний опис метрик: python main.py --metrics-help\n"
        )

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
            if verbose:
                traj = entry.get("trajectory") or {}
                gp = traj.get("grounding_passed")
                print(f"           grounding_passed={gp!r}")
                for line in entry.get("reasons") or []:
                    print(f"           + {line}")
                for line in entry.get("demerits") or []:
                    print(f"           - {line}")

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
        if t:
            tool_counts[t] = tool_counts.get(t, 0) + 1

    avg_latency = sum(r.get("elapsed_s", 0) for r in results) / max(len(results), 1)
    avg_tools   = sum(
        r.get("trajectory", {}).get("tool_calls", 0) for r in results
    ) / max(len(results), 1)

    # ── Додаткові метрики для консолі / summary.json ─────────────────
    error_tasks = sum(1 for r in results if r.get("error"))
    g_true = g_false = g_none = 0
    tok_in = tok_out = 0
    for r in results:
        traj = r.get("trajectory") or {}
        gp = traj.get("grounding_passed")
        if gp is True:
            g_true += 1
        elif gp is False:
            g_false += 1
        else:
            g_none += 1
        est = traj.get("token_usage_est") or {}
        tok_in += int(est.get("input_tokens_est") or 0)
        tok_out += int(est.get("output_tokens_est") or 0)
    g_decided = g_true + g_false
    grounding_pass_rate = round(g_true / max(g_decided, 1), 3) if g_decided else None

    task_by_id = {t["task_id"]: t for t in tasks}
    triage_metrics = aggregate_triage_metrics(results, task_by_id)

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
        "error_tasks":    error_tasks,
        "grounding": {
            "passed_true":  g_true,
            "passed_false": g_false,
            "unknown":      g_none,
            "pass_rate_decided": grounding_pass_rate,
        },
        "token_usage_est_total": {
            "input_tokens":  tok_in,
            "output_tokens": tok_out,
            "total_tokens":  tok_in + tok_out,
        },
        "triage_metrics": triage_metrics,
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
    if error_tasks:
        print(f"  Errors:  {error_tasks}/{len(tasks)} tasks crashed")
    print(
        f"  Grounding: {g_true} true / {g_false} false / {g_none} n/a "
        f"(among tasks with report: {g_decided} decided)"
    )
    if grounding_pass_rate is not None:
        print(f"             pass rate (true / decided): {grounding_pass_rate*100:.0f}%")
    print(
        f"  Tokens (est.): in={tok_in:,}  out={tok_out:,}  total={tok_in + tok_out:,}"
    )

    tm = triage_metrics or {}
    if tm:
        print(f"\n  Triage evaluation:")
        print(
            f"    Perfect score (max pts): {tm.get('perfect_tasks', 0)}/{tm.get('non_crash_tasks', 0)} "
            f"({(tm.get('perfect_rate') or 0)*100:.0f}% of non-crashed)"
        )
        print(f"    Score histogram (points→count): {tm.get('score_histogram', {})}")
        gw = tm.get("grounding_when_required") or {}
        if gw.get("tasks"):
            gr = gw.get("pass_rate")
            grs = f"{float(gr)*100:.0f}%" if gr is not None else "n/a"
            print(
                f"    Grounding when required: {gw.get('grounded', 0)}/{gw.get('tasks', 0)} ({grs})"
            )
        inc = tm.get("passed_but_grounding_failed", 0)
        if inc:
            print(f"    WARN: pass threshold but grounding failed (required): {inc} tasks")
        rs = tm.get("rubric_signals") or {}
        print(
            f"    Rubric: avg reasons/task={rs.get('avg_reasons_per_task')}, "
            f"avg demerits/task={rs.get('avg_demerits_per_task')}"
        )
        print(f"\n  By task type:")
        for typ, s in sorted((tm.get("by_task_type") or {}).items()):
            pct = s["pass"] / max(s["total"], 1) * 100
            print(f"    {typ:16s} {s['pass']}/{s['total']} pass ({pct:.0f}%) — {s['points']}/{s['max']}pt")

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
    parser.add_argument("-v", "--verbose", action="store_true", help="Друкувати reasons/demerits по задачі")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        output_dir=args.output,
        tasks_filter=args.tasks,
        model=args.model,
        verbose=args.verbose,
    ))
