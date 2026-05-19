#!/usr/bin/env python3
"""
Збирає зведення з evaluation/summary.json і показує метрики.

  # Після повного eval:
  python main.py --eval --output trajectories
  python scripts/show_eval_summary.py

  # Після eval — зведення + по кожній задачі:
  python scripts/show_eval_summary.py --per-task

  # Повний JSON (як раніше):
  python scripts/show_eval_summary.py --json
  python scripts/show_eval_summary.py trajectories/summary.json --json

  # Інший summary / tasks.json:
  python scripts/show_eval_summary.py path/to/summary.json --per-task --tasks path/to/tasks.json

  # Два відокремлені блоки (зведення + по кожній задачі, як після eval):
  python scripts/show_eval_summary.py --eval-blocks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_JSON = ROOT / "evaluation" / "tasks.json"

W = 62


def _grounding_from_result(r: dict) -> bool | None:
    """Як у eval: спочатку state (якщо є), інакше trajectory."""
    st = r.get("state") or {}
    gp = st.get("grounding_passed")
    if gp is not None:
        return bool(gp)
    traj = r.get("trajectory") or {}
    gp2 = traj.get("grounding_passed")
    if gp2 is None:
        return None
    return bool(gp2)


def print_per_task_lines(data: dict, tasks_path: Path) -> None:
    """Один блок на кожну задачу з results[] (потрібен eval, що пише results у summary)."""
    results = data.get("results") or []
    if not results:
        print("\n  Немає results у summary.json — перезапустіть eval (python main.py --eval ...).")
        return

    task_by_id: dict = {}
    if tasks_path.is_file():
        task_by_id = {t["task_id"]: t for t in json.loads(tasks_path.read_text(encoding="utf-8"))}

    print("\n" + "═" * W)
    print("  ПО КОЖНОМУ ТРІАЖУ (з results + tasks.json)")
    print("═" * W)

    for i, r in enumerate(results, 1):
        tid = r.get("task_id", "?")
        task = task_by_id.get(tid, {})
        typ = r.get("type") or task.get("type") or "?"
        cat = r.get("category") or task.get("category") or "?"
        sc = int(r.get("score") or 0)
        mx = int(r.get("max_points") or 0)
        err = r.get("error")

        crit = task.get("success_criteria") or {}
        g_req = bool(crit.get("grounding_required"))
        gp = _grounding_from_result(r)
        if gp is True:
            g_s = "true"
        elif gp is False:
            g_s = "false"
        else:
            g_s = "n/a"

        traj = r.get("trajectory") or {}

        if err:
            print(f"\n  [{i:02d}] {tid}  type={typ}  cat={cat}")
            print(f"       ERROR: {err}")
            if r.get("elapsed_s") is not None:
                print(f"       elapsed_s={r.get('elapsed_s')}")
            continue

        passed = bool(r.get("passed"))
        perfect = mx > 0 and sc == mx
        reasons = r.get("reasons") or []
        demerits = r.get("demerits") or []
        warn_g = ""
        if g_req and passed and gp is not True:
            warn_g = "  [WARN] pass, але grounding не OK (grounding_required)"

        st_pass = "PASS" if passed else "FAIL"
        print(f"\n  [{i:02d}] {tid}  type={typ}  cat={cat}")
        print(
            f"       {st_pass}  {sc}/{mx} pt  perfect={str(perfect).lower()}  "
            f"grounding_required={str(g_req).lower()}  grounding_passed={g_s}  "
            f"predicted_type={r.get('predicted_classification')!r}"
        )
        est = traj.get("token_usage_est") or {}
        tin = int(est.get("input_tokens_est") or 0)
        tout = int(est.get("output_tokens_est") or 0)
        print(
            f"       elapsed_s={r.get('elapsed_s', '—')}  tool_calls={traj.get('tool_calls', '—')}  "
            f"tokens_est in={tin:,} out={tout:,} total={tin + tout:,}"
        )
        print(f"       reasons={len(reasons)}  demerits={len(demerits)}{warn_g}")
        for line in reasons:
            print(f"         + {line}")
        for line in demerits:
            print(f"         - {line}")

    print("\n" + "═" * W + "\n")


def print_triage_metrics_full(tm: dict | None) -> None:
    """Усі поля aggregate `triage_metrics` з evaluation/runner.py."""
    if tm is None:
        print("\n  Оцінка тріажу (aggregate):")
        print("    У summary немає ключа triage_metrics — перезапустіть eval (python main.py --eval).")
        return
    if len(tm) == 0:
        print("\n  Оцінка тріажу (aggregate):")
        print("    triage_metrics порожній (0 задач у прогоні або застарілий формат).")
        return
    print("\n  Оцінка тріажу (aggregate, усі поля):")
    print(f"    non_crash_tasks:             {tm.get('non_crash_tasks')}")
    print(f"    crashed_tasks:               {tm.get('crashed_tasks')}")
    print(f"    perfect_tasks:               {tm.get('perfect_tasks')}")
    pr = tm.get("perfect_rate")
    if isinstance(pr, (int, float)):
        print(f"    perfect_rate:                {pr}  ({pr * 100:.1f}% від non_crash)")
    else:
        print(f"    perfect_rate:                {pr}")
    print(f"    score_histogram:             {tm.get('score_histogram', {})}")
    gw = tm.get("grounding_when_required") or {}
    gtasks = gw.get("tasks", 0)
    gok = gw.get("grounded", 0)
    graw = gw.get("pass_rate")
    if gtasks:
        grs = f"{float(graw) * 100:.1f}%" if graw is not None else "—"
        print(f"    grounding_when_required:     задач={gtasks}  grounded_ok={gok}  pass_rate={graw} ({grs})")
    else:
        print("    grounding_when_required:     задач з grounding_required=0  (pass_rate n/a)")
    pbgf = tm.get("passed_but_grounding_failed", 0)
    print(f"    passed_but_grounding_failed: {pbgf}")
    if pbgf:
        print("      (pass за порогом, але grounding не true, де success_criteria.grounding_required)")
    rs = tm.get("rubric_signals") or {}
    print("    rubric_signals:")
    print(f"      avg_reasons_per_task:      {rs.get('avg_reasons_per_task')}")
    print(f"      avg_demerits_per_task:     {rs.get('avg_demerits_per_task')}")
    print(f"      total_pass_reasons:        {rs.get('total_pass_reasons')}")
    print(f"      total_demerits:            {rs.get('total_demerits')}")
    lat = tm.get("latency_s") or {}
    if lat.get("mean") is not None:
        print("    latency_s:")
        print(
            f"      mean={lat.get('mean')}  p50={lat.get('p50')}  p90={lat.get('p90')}  p95={lat.get('p95')}  "
            f"min={lat.get('min')}  max={lat.get('max')}  stdev={lat.get('stdev')}"
        )
    cp = tm.get("conditional_pass") or {}
    if cp:
        print(f"    conditional_pass:            {cp}")
    gvs = tm.get("grounding_vs_score") or {}
    if gvs:
        print(f"    grounding_vs_score:          {gvs}")
    tcm = tm.get("type_confusion_matrix") or []
    if tcm:
        print("    type_confusion_matrix:")
        for row in tcm[:15]:
            print(f"      {row}")
        if len(tcm) > 15:
            print(f"      … +{len(tcm) - 15} rows")
    byt = tm.get("by_task_type") or {}
    print("    by_task_type:")
    if byt:
        for typ, s in sorted(byt.items()):
            tot = s.get("total", 0)
            ok = s.get("pass", 0)
            pts = s.get("points", 0)
            mx = s.get("max", 0)
            p = ok / tot * 100 if tot else 0
            print(f"      • {typ:16s}  {ok}/{tot} pass ({p:.0f}%)   {pts}/{mx} pt")
    else:
        print("      (порожньо)")


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

    if "error_tasks" in data:
        err = int(data.get("error_tasks") or 0)
        print(f"  Помилки (crash): {err}/{n} задач")

    g = data.get("grounding")
    if isinstance(g, dict):
        gt, gf, gu = g.get("passed_true", 0), g.get("passed_false", 0), g.get("unknown", 0)
        decided = gt + gf
        print(f"  Grounding:       {gt} true / {gf} false / {gu} n/a (decided={decided})")
        prg = g.get("pass_rate_decided")
        if prg is not None and decided:
            print(f"                    true/decided: {float(prg)*100:.0f}%")
        elif prg is not None:
            print("                    true/decided: n/a (немає decided)")

    tok = data.get("token_usage_est_total")
    if isinstance(tok, dict):
        ti, to = int(tok.get("input_tokens") or 0), int(tok.get("output_tokens") or 0)
        print(f"  Токени (оцінка): in={ti:,}  out={to:,}  total={ti + to:,}")

    tm_raw = data.get("triage_metrics")
    print_triage_metrics_full(tm_raw if isinstance(tm_raw, dict) else None)

    if isinstance(tm_raw, dict):
        ex = tm_raw.get("extended_metrics_summary")
        if isinstance(ex, dict) and ex.get("tasks_with_extended_metrics"):
            print("\n  Розширені метрики (середнє по eval):")
            print(f"    completeness: {ex.get('avg_report_completeness')}  actionability: {ex.get('avg_actionability')}")
            print(f"    hallucination_rate: {ex.get('hallucination_rate')}  avg_search_queries: {ex.get('avg_search_query_count')}")
            if ex.get("type_correct_rate") is not None:
                print(f"    type_correct_rate: {ex.get('type_correct_rate')} ({ex.get('type_correct_evaluated')} tasks)")

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
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Після зведення: деталі по кожній задачі (results у summary + evaluation/tasks.json)",
    )
    parser.add_argument(
        "--tasks",
        default=str(TASKS_JSON),
        help=f"Шлях до tasks.json (за замовч.: {TASKS_JSON})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Як --per-task: зведення + повний aggregate triage + деталі по кожній задачі",
    )
    parser.add_argument(
        "--eval-blocks",
        action="store_true",
        help="Додати блоки EVAL: зведення без results[] і JSON по кожній задачі (рубрика + trajectory)",
    )
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
        if args.per_task or args.all:
            print_per_task_lines(data, Path(args.tasks).resolve())
    if args.eval_blocks:
        from evaluation.runner import print_eval_metrics_separated

        print_eval_metrics_separated(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
