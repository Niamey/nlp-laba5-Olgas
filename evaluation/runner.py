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
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.issue_types import normalize_issue_type

logger = logging.getLogger("eval")


TASKS_FILE = Path(__file__).parent / "tasks.json"


def _strip_message_trace_nested(obj: object) -> object:
    """Прибирає message_trace з dict/list для компактного друку в консолі."""
    if isinstance(obj, dict):
        return {
            k: _strip_message_trace_nested(v)
            for k, v in obj.items()
            if k != "message_trace"
        }
    if isinstance(obj, list):
        return [_strip_message_trace_nested(x) for x in obj]
    return obj


def print_eval_metrics_separated(summary: dict) -> None:
    """
    Окремі блоки в консолі: (1) зведення eval без results[], (2) по кожній задачі —
    рубрика + зріз trajectory (як у T00x.json; повний звіт — full_report у файлі).
    """
    sep = "═" * 62
    results = summary.get("results") or []

    print(f"\n{sep}")
    print("  БЛОК EVAL — зведення по всьому прогону (≈ summary.json без results[])")
    print(sep)
    overview = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(overview, indent=2, ensure_ascii=False, default=str))

    print(f"\n{sep}")
    print(
        "  БЛОК EVAL — по кожній задачі: score, rubric (reasons/demerits), "
        "predicted_classification, elapsed_s, trajectory (без message_trace), report_snippet"
    )
    print(f"  Повний звіт у файлах: trajectories/T00x.json → full_report")
    print(sep)
    for r in results:
        slim = {
            "task_id":                  r.get("task_id"),
            "category":                 r.get("category"),
            "type":                     r.get("type"),
            "score":                    r.get("score"),
            "max_points":               r.get("max_points"),
            "passed":                   r.get("passed"),
            "predicted_classification": r.get("predicted_classification"),
            "elapsed_s":                r.get("elapsed_s"),
            "error":                    r.get("error"),
            "reasons":                  r.get("reasons"),
            "demerits":                 r.get("demerits"),
            "report_snippet":           r.get("report_snippet"),
            "trajectory":               _strip_message_trace_nested(r.get("trajectory") or {}),
        }
        print(json.dumps(slim, indent=2, ensure_ascii=False, default=str))
        print()

    print(f"{sep}\n")


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


def _type_correct_allowed_set(correct_types: list | tuple) -> set[str]:
    allowed: set[str] = set()
    for x in correct_types:
        allowed.add(str(x).lower().strip())
    if "docs" in allowed:
        allowed.add("documentation")
    if "documentation" in allowed:
        allowed.add("docs")
    return allowed


def _fallback_type_from_report(report: str) -> str:
    """Якщо воркер не поклав classification у state — пробуємо витягти type з тексту звіту."""
    if not report:
        return ""
    lo = report.lower()
    m = re.search(r'"type"\s*:\s*"(bug|feature|question|documentation|duplicate)"', lo)
    if m:
        return m.group(1)
    m2 = re.search(
        r"\*\*labels?\*\*[^\n]*\b(bug|feature|question|documentation|duplicate)\b",
        lo,
    )
    if m2:
        return m2.group(1)
    return ""


def _type_correct_structural_match(
    allowed: set[str],
    pred_norm: str,
    report: str,
) -> bool:
    if pred_norm and pred_norm in allowed:
        return True
    if pred_norm == "bug" and "performance" in allowed:
        if any(k in report for k in ("performance", "slow", "latency", "memory", "cpu")):
            return True
    if pred_norm == "bug" and "security" in allowed:
        if any(k in report for k in ("security", "cve", "vulnerability", "auth", "injection", "exploit")):
            return True
    return False


def aggregate_triage_metrics(results: list[dict], task_by_id: dict[str, dict]) -> dict:
    """
    Додаткові метрики оцінки тріажу (поруч із score/pass): розподіл балів, тип задачі,
    grounding там де він вимагався рубрикою, сигнали rubric (reasons/demerits).
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


def _aggregate_fact_check(results: list[dict]) -> dict:
    """
    Зведення по regex fact-checker (agent/fact_extractor.py).
    Підраховує: задачі з фактами, скільки grounded/invented по типах,
    середній facts_grounded_rate.
    """
    tasks_with_facts = 0
    total_facts = 0
    total_grounded = 0
    total_invented = 0
    by_kind: dict[str, int] = {"issue_numbers": 0, "usernames": 0, "file_paths": 0, "urls": 0}
    fail_count = 0
    pass_count = 0
    rate_sum = 0.0
    rate_n = 0
    invented_examples: list[dict] = []

    for r in results:
        if r.get("error"):
            continue
        traj = r.get("trajectory") or {}
        fc = traj.get("fact_check")
        if not isinstance(fc, dict):
            continue
        tf = int(fc.get("total_facts") or 0)
        if tf <= 0:
            continue
        tasks_with_facts += 1
        total_facts += tf
        total_grounded += int(fc.get("grounded_facts") or 0)
        invented = fc.get("invented") or []
        total_invented += len(invented)
        for item in invented:
            kind = (item or {}).get("kind")
            if kind in by_kind:
                by_kind[kind] += 1
            if len(invented_examples) < 8:
                invented_examples.append({
                    "task_id": r.get("task_id"),
                    "kind":    kind,
                    "value":   (item or {}).get("value"),
                })
        rate = fc.get("facts_grounded_rate")
        if isinstance(rate, (int, float)):
            rate_sum += float(rate)
            rate_n += 1
        if fc.get("passed"):
            pass_count += 1
        else:
            fail_count += 1

    avg_rate = round(rate_sum / rate_n, 3) if rate_n else None
    overall_rate = round(total_grounded / max(total_facts, 1), 3) if total_facts else None
    return {
        "tasks_with_facts":      tasks_with_facts,
        "total_facts_extracted": total_facts,
        "total_grounded":        total_grounded,
        "total_invented":        total_invented,
        "invented_by_kind":      by_kind,
        "pass":                  pass_count,
        "fail":                  fail_count,
        "avg_facts_grounded_rate": avg_rate,
        "overall_grounded_rate":   overall_rate,
        "invented_examples":     invented_examples,
    }


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    """q у [0,1], напр. 0.95 для p95."""
    if not sorted_vals:
        return None
    xs = sorted(sorted_vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


def _stdev(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return round(var**0.5, 2)


def build_auxiliary_metrics(results: list[dict], task_by_id: dict[str, dict]) -> dict:
    """
    Додаткові метрики: latency percentiles, умовні pass-рейти, матриця типів (gold vs predicted),
    зв'язок grounding ↔ score.
    """
    lat_ok: list[float] = []
    must_search_n = must_search_ok = 0
    type_correct_n = type_correct_ok = 0
    from collections import Counter

    confusion: Counter[tuple[str, str]] = Counter()
    scores_if_g_true: list[int] = []
    scores_if_g_false: list[int] = []
    hist_false: dict[str, int] = {}
    scores_req_g_true: list[int] = []
    scores_req_g_false: list[int] = []

    for r in results:
        if r.get("error"):
            continue
        es = r.get("elapsed_s")
        if es is not None:
            try:
                lat_ok.append(float(es))
            except (TypeError, ValueError):
                pass

        tid = r.get("task_id") or ""
        task = task_by_id.get(tid, {})
        crit = task.get("success_criteria") or {}

        if crit.get("must_search"):
            must_search_n += 1
            if r.get("passed"):
                must_search_ok += 1

        tc = crit.get("type_correct")
        if tc:
            if not isinstance(tc, (list, tuple)):
                tc = [tc]
            gold_key = "+".join(sorted({str(x).lower() for x in tc}))
            pred = (r.get("predicted_classification") or "").strip().lower() or "?"
            confusion[(gold_key, pred)] += 1
            type_correct_n += 1
            reasons = r.get("reasons") or []
            if any("PASS: Correct type detected" in str(x) for x in reasons):
                type_correct_ok += 1

        traj = r.get("trajectory") or {}
        gp = traj.get("grounding_passed")
        sc = int(r.get("score") or 0)
        if gp is True:
            scores_if_g_true.append(sc)
        elif gp is False:
            scores_if_g_false.append(sc)
            k = str(sc)
            hist_false[k] = hist_false.get(k, 0) + 1

        if crit.get("grounding_required"):
            if gp is True:
                scores_req_g_true.append(sc)
            elif gp is False:
                scores_req_g_false.append(sc)

    latency_block: dict = {}
    if lat_ok:
        latency_block = {
            "mean":   round(sum(lat_ok) / len(lat_ok), 2),
            "p50":    round(_percentile(lat_ok, 0.5) or 0, 2),
            "p90":    round(_percentile(lat_ok, 0.9) or 0, 2),
            "p95":    round(_percentile(lat_ok, 0.95) or 0, 2),
            "min":    round(min(lat_ok), 2),
            "max":    round(max(lat_ok), 2),
            "stdev":  _stdev(lat_ok),
            "tasks":  len(lat_ok),
        }

    ms_rate = round(must_search_ok / max(must_search_n, 1), 3) if must_search_n else None
    tc_rate = round(type_correct_ok / max(type_correct_n, 1), 3) if type_correct_n else None

    confusion_list = [
        {"gold": g, "predicted": p, "count": c}
        for (g, p), c in sorted(confusion.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]

    m_gt = round(sum(scores_if_g_true) / len(scores_if_g_true), 2) if scores_if_g_true else None
    m_gf = round(sum(scores_if_g_false) / len(scores_if_g_false), 2) if scores_if_g_false else None
    m_req_gt = round(sum(scores_req_g_true) / len(scores_req_g_true), 2) if scores_req_g_true else None
    m_req_gf = round(sum(scores_req_g_false) / len(scores_req_g_false), 2) if scores_req_g_false else None

    fact_summary = _aggregate_fact_check(results)

    return {
        "latency_s": latency_block,
        "fact_check_summary": fact_summary,
        "conditional_pass": {
            "must_search": {
                "tasks": must_search_n,
                "pass":  must_search_ok,
                "rate":  ms_rate,
            },
            "type_correct_rubric": {
                "tasks": type_correct_n,
                "pass":  type_correct_ok,
                "rate":  tc_rate,
            },
        },
        "type_confusion_matrix": confusion_list,
        "grounding_vs_score": {
            "n_grounding_true":          len(scores_if_g_true),
            "mean_score_grounding_true": m_gt,
            "n_grounding_false":         len(scores_if_g_false),
            "mean_score_grounding_false": m_gf,
            "score_histogram_when_grounding_false": hist_false,
            "subset_grounding_required_only": {
                "n_true":          len(scores_req_g_true),
                "mean_true":       m_req_gt,
                "n_false":         len(scores_req_g_false),
                "mean_false":      m_req_gf,
            },
        },
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

    if criteria.get("must_contain"):
        phrases = criteria["must_contain"]
        if not isinstance(phrases, (list, tuple)):
            phrases = [phrases]
        missing = [str(p) for p in phrases if str(p).strip().lower() not in report]
        if missing:
            demerits.append(f"FAIL: Report must mention (substring): {missing}")
            score = max(0, score - 1)

    if criteria.get("must_report_not_found"):
        if "not found" in report or "404" in report or "does not exist" in report:
            reasons.append("PASS: Correctly reported not found")
        else:
            demerits.append("FAIL: Did not report not found")
            score = max(0, score - 2)

    # Grounding: завжди штраф, якщо валідатор не підтвердив (узгодження з trajectory.grounding_passed).
    if grounding_ok is True:
        if criteria.get("grounding_required"):
            reasons.append("PASS: Grounding check passed")
    elif result.get("report"):
        if criteria.get("grounding_required"):
            demerits.append("WARN: Grounding check failed (required)")
        else:
            demerits.append("WARN: Grounding validation failed (report vs tool evidence)")
        score = max(0, score - 1)

    tools_called = [r.get("tool") for r in trajectory.get("tool_results", [])]
    search_calls = [r for r in trajectory.get("tool_results", []) if r.get("tool") == "search_similar_issues"]

    if criteria.get("must_search"):
        if "search_similar_issues" in tools_called:
            reasons.append("PASS: Searched for similar issues")
        else:
            demerits.append("FAIL: Did not search for similar issues")
            score = max(0, score - 2)

    min_q = criteria.get("min_search_queries")
    if min_q is not None:
        need = int(min_q)
        got = len(search_calls)
        if got >= need:
            reasons.append(f"PASS: At least {need} search query/queries ({got})")
        else:
            demerits.append(f"FAIL: Expected at least {need} search queries, got {got}")
            score = max(0, score - 1)

    if criteria.get("must_flag_security"):
        sec_kw = ("security", "cve", "vulnerability", "exploit", "xss", "injection", "auth bypass")
        if any(k in report for k in sec_kw):
            reasons.append("PASS: Security aspect flagged in report")
        else:
            demerits.append("FAIL: Security issue not flagged in report")
            score = max(0, score - 1)

    if criteria.get("must_compare_root_causes"):
        rc_kw = (
            "root cause", "root-cause", "compare", "versus", "same underlying",
            "different cause", "not a duplicate", "related but",
        )
        if any(k in report for k in rc_kw):
            reasons.append("PASS: Root cause comparison present")
        else:
            demerits.append("FAIL: No root cause comparison in report")
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
        if not isinstance(correct_types, (list, tuple)):
            correct_types = [correct_types]
        allowed = _type_correct_allowed_set(correct_types)

        findings = (state.get("triage_findings") or {}).get("classification")
        raw_pred = ""
        if isinstance(findings, dict):
            raw_pred = str(findings.get("type") or "")
        pred_norm = normalize_issue_type(raw_pred)

        ok_struct = _type_correct_structural_match(allowed, pred_norm, report)
        ok_report = any(str(t).lower() in report for t in correct_types)

        if ok_struct or ok_report:
            reasons.append(f"PASS: Correct type detected ({correct_types})")
        else:
            demerits.append(
                f"FAIL: Expected type in {list(allowed)} "
                f"(structured={pred_norm or '—'}, report_substring_match={ok_report})"
            )
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


def _effective_user_prompt(task: dict, default_prompt: str | None) -> str | None:
    from agent.user_instructions import effective_user_prompt_for_task

    return effective_user_prompt_for_task(task, default_prompt)


async def run_evaluation(
    output_dir:   str = "trajectories",
    tasks_filter: Optional[list] = None,
    model:        Optional[str]  = None,
    no_hil:       bool = True,
    user_prompt:  Optional[str] = None,
):
    """
    Запускає всі задачі з tasks.json і зберігає результати.

    Args:
        output_dir:   директорія для trajectory JSON файлів
        tasks_filter: список task_id для запуску (None = всі)
        model:        override PRIMARY_MODEL env var
        no_hil:       True = вимкнути HIL для batch evaluation
        user_prompt:  інструкції для кожної задачі (CLI --prompt / --prompt-file);
                      перекривається input.user_prompt у tasks.json для окремої задачі
    """
    import os
    if model:
        os.environ["PRIMARY_MODEL"] = model

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
    if user_prompt:
        preview = user_prompt.replace("\n", " ")[:72]
        print(f"  Default user prompt: {len(user_prompt)} chars — {preview!r}…")
    print(f"{'═'*60}\n")

    for i, task in enumerate(tasks, 1):
        task_id  = task["task_id"]
        category = task.get("category", "?")
        task_prompt = _effective_user_prompt(task, user_prompt)
        prompt_note = f" | prompt={len(task_prompt)}ch" if task_prompt else ""
        print(f"[{i:02d}/{len(tasks)}] {task_id} ({category}) — {task['description'][:50]}...{prompt_note}")

        start = time.time()
        try:
            result = await run_triage(
                issue_url=task["input"]["issue_url"],
                task_hint=task["input"].get("hint"),
                user_prompt=task_prompt,
                human_in_loop=not no_hil,
            )
            elapsed = round(time.time() - start, 2)

            score_result_data = await score_result(task, result)

            state_obj = result.get("state") or {}
            findings = state_obj.get("triage_findings") or {}
            cls = findings.get("classification") if isinstance(findings.get("classification"), dict) else {}
            raw_p = (cls.get("type") or "").strip() if cls else ""
            predicted_norm = normalize_issue_type(raw_p)
            if not predicted_norm:
                predicted_norm = normalize_issue_type(_fallback_type_from_report(result.get("report") or ""))
            predicted_cls = predicted_norm or None

            from agent.extended_metrics import compute_extended_metrics

            traj = result.get("trajectory", {})
            extended = compute_extended_metrics(
                report=result.get("report", ""),
                state=result.get("state") or {},
                trajectory=traj,
                task=task,
            )

            entry = {
                "task_id":    task_id,
                "category":   category,
                "type":       task.get("type"),
                "user_prompt": task_prompt,
                "score":      score_result_data["score"],
                "max_points": score_result_data["max_points"],
                "passed":     score_result_data["passed"],
                "reasons":    score_result_data["reasons"],
                "demerits":   score_result_data["demerits"],
                "trajectory": traj,
                "extended_metrics": extended,
                "report_snippet": result.get("report", "")[:300],
                "elapsed_s":  elapsed,
                "error":      None,
                "predicted_classification": predicted_cls,
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

    task_by_id = {t["task_id"]: t for t in tasks}
    triage_metrics = aggregate_triage_metrics(results, task_by_id)
    triage_metrics.update(build_auxiliary_metrics(results, task_by_id))
    from agent.extended_metrics import aggregate_extended_metrics_summary

    triage_metrics["extended_metrics_summary"] = aggregate_extended_metrics_summary(results)

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

    summary = {
        "run_timestamp":  run_ts,
        "model":          model or os.getenv("PRIMARY_MODEL", "default"),
        "user_prompt_default": user_prompt,
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
    lb = (triage_metrics or {}).get("latency_s") or {}
    if lb:
        print(
            f"           p50={lb.get('p50')}s  p90={lb.get('p90')}s  p95={lb.get('p95')}s  "
            f"min={lb.get('min')}s  max={lb.get('max')}s  stdev={lb.get('stdev')}"
        )
    print(f"  Tools:   {avg_tools:.1f} avg per task")
    if error_tasks:
        print(f"  Errors:  {error_tasks}/{len(tasks)} tasks crashed")
    print(
        f"  Grounding: {g_true} true / {g_false} false / {g_none} n/a "
        f"(decided={g_decided})"
    )
    if grounding_pass_rate is not None:
        print(f"             true/decided: {grounding_pass_rate*100:.0f}%")
    print(
        f"  Tokens (est.): in={tok_in:,}  out={tok_out:,}  total={tok_in + tok_out:,}"
    )
    if tok_in + tok_out == 0:
        print("             (if zero: ensure LLM nodes use aimessage_from_llm — see agent/llm_messages.py)")

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
        cp = tm.get("conditional_pass") or {}
        ms = cp.get("must_search") or {}
        if ms.get("tasks"):
            rms = ms.get("rate")
            rms_pct = f"{float(rms)*100:.0f}%" if rms is not None else "n/a"
            print(f"    Pass | must_search tasks: {ms.get('pass', 0)}/{ms.get('tasks', 0)} ({rms_pct})")
        tc = cp.get("type_correct_rubric") or {}
        if tc.get("tasks"):
            rtc = tc.get("rate")
            rtc_pct = f"{float(rtc)*100:.0f}%" if rtc is not None else "n/a"
            print(f"    Pass | type_correct tasks: {tc.get('pass', 0)}/{tc.get('tasks', 0)} ({rtc_pct})")
        fcs = tm.get("fact_check_summary") or {}
        if fcs.get("tasks_with_facts"):
            r_ov = fcs.get("overall_grounded_rate")
            r_av = fcs.get("avg_facts_grounded_rate")
            r_ov_s = f"{float(r_ov)*100:.0f}%" if r_ov is not None else "n/a"
            r_av_s = f"{float(r_av)*100:.0f}%" if r_av is not None else "n/a"
            print(
                f"    Fact-check (regex): {fcs.get('pass', 0)} pass / {fcs.get('fail', 0)} fail "
                f"(tasks_with_facts={fcs.get('tasks_with_facts')}); "
                f"overall_rate={r_ov_s} avg_rate={r_av_s}"
            )
            ik = fcs.get("invented_by_kind") or {}
            print(
                f"      invented: issues={ik.get('issue_numbers', 0)} "
                f"users={ik.get('usernames', 0)} "
                f"paths={ik.get('file_paths', 0)} "
                f"urls={ik.get('urls', 0)} "
                f"(total={fcs.get('total_invented', 0)} / {fcs.get('total_facts_extracted', 0)} facts)"
            )
            ex = fcs.get("invented_examples") or []
            if ex:
                preview = "; ".join(
                    f"{x.get('task_id')}:{x.get('kind')}={x.get('value')!r}" for x in ex[:5]
                )
                print(f"      examples: {preview}")

        gvs = tm.get("grounding_vs_score") or {}
        if gvs.get("n_grounding_false", 0) or gvs.get("n_grounding_true", 0):
            print(
                f"    Grounding vs score: true n={gvs.get('n_grounding_true')} "
                f"mean_score={gvs.get('mean_score_grounding_true')}; "
                f"false n={gvs.get('n_grounding_false')} mean_score={gvs.get('mean_score_grounding_false')} "
                f"histogram_if_false={gvs.get('score_histogram_when_grounding_false', {})}"
            )
            sub = gvs.get("subset_grounding_required_only") or {}
            if sub.get("n_true") or sub.get("n_false"):
                print(
                    f"      subset grounding_required: true n={sub.get('n_true')} mean={sub.get('mean_true')}; "
                    f"false n={sub.get('n_false')} mean={sub.get('mean_false')}"
                )
        tcm = tm.get("type_confusion_matrix") or []
        if tcm:
            print(f"\n  Type confusion (gold from tasks.json → predicted from worker):")
            for row in tcm[:12]:
                print(f"    {row.get('gold')!s:20s} → {row.get('predicted')!s:16s}  n={row.get('count')}")
            if len(tcm) > 12:
                print(f"    … +{len(tcm) - 12} more rows")
        print(f"\n  By task type (from tasks.json):")
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

    print_eval_metrics_separated(summary)

    return summary


if __name__ == "__main__":
    import argparse

    from agent.prompt_quality import validate_user_prompt
    from agent.user_instructions import load_user_prompt_from_cli

    parser = argparse.ArgumentParser(description="Eval runner (tasks.json)")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run (T001 …)")
    parser.add_argument("--model", help="Override PRIMARY_MODEL")
    parser.add_argument("--output", default="trajectories")
    parser.add_argument("--prompt", help="User instructions for every task (unless input.user_prompt in JSON)")
    parser.add_argument("--prompt-file", metavar="PATH", help="UTF-8 file; overrides --prompt")
    args = parser.parse_args()

    up = load_user_prompt_from_cli(args.prompt, args.prompt_file)
    if up:
        v = validate_user_prompt(up)
        if not v.is_valid:
            for issue in v.issues:
                print(f"✗ {issue}")
            raise SystemExit(2)
        up = v.normalized

    asyncio.run(run_evaluation(
        output_dir=args.output,
        tasks_filter=args.tasks,
        model=args.model,
        user_prompt=up,
    ))
