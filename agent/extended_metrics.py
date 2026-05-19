"""
Розширені метрики одного прогону тріажу (якість звіту, tools, галюцинації, вартість).

Використання:
  from agent.extended_metrics import compute_extended_metrics, print_extended_metrics

  metrics = compute_extended_metrics(report=..., state=..., trajectory=..., task=...)
  print_extended_metrics(metrics, issue_url=...)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# ── Report sections (markdown headings) ─────────────────────────────
_RE_FLAGS = re.IGNORECASE | re.MULTILINE
_SECTION_PATTERNS = {
    "summary": re.compile(
        r"^\s*#+\s*\*?\*?summary\*?\*?\s*$|^\s*\*\*summary\*\*",
        _RE_FLAGS,
    ),
    "analysis": re.compile(
        r"^\s*#+\s*\*?\*?(analysis|root\s*cause)\*?\*?\s*$|^\s*\*\*(analysis|root\s*cause)\*\*",
        _RE_FLAGS,
    ),
    "recommendation": re.compile(
        r"^\s*#+\s*\*?\*?(recommendation|next\s*steps?)\*?\*?\s*$|^\s*\*\*recommendation\*\*",
        _RE_FLAGS,
    ),
    "labels": re.compile(
        r"^\s*#+\s*\*?\*?labels?\*?\*?\s*$|^\s*\*\*labels?\*\*",
        _RE_FLAGS,
    ),
}

_ACTIONABILITY_KW = (
    "recommend", "suggest", "should", "close", "merge", "label", "assign",
    "duplicate of", "investigate", "fix", "ping", "next step", "action",
)

_SECURITY_KW = (
    "security", "cve", "vulnerability", "exploit", "xss", "injection",
    "auth bypass", "privilege", "malicious",
)

_ROOT_CAUSE_KW = (
    "root cause", "root-cause", "compare", "versus", "same underlying",
    "different cause", "not a duplicate", "related but",
)

_CONFIDENCE_MAP = {"high": 1.0, "medium": 0.66, "low": 0.33}


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x or "")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _tool_call_signature(rec: dict) -> str:
    tool = rec.get("tool", "?")
    args = rec.get("args")
    if args is None:
        return tool
    try:
        return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        return f"{tool}:{args!r}"


def _tool_error(rec: dict) -> bool:
    res = rec.get("result")
    if isinstance(res, dict) and res.get("error"):
        return True
    if isinstance(res, str) and ("error" in res.lower()[:200] or "failed" in res.lower()[:80]):
        return True
    return False


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float | None:
    """Груба оцінка за env або дефолт (USD за 1M токенів)."""
    try:
        pin = float(os.getenv("METRICS_COST_INPUT_PER_1M", "1.25"))
        pout = float(os.getenv("METRICS_COST_OUTPUT_PER_1M", "5.0"))
    except ValueError:
        return None
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 6)


def compute_report_quality(report: str) -> dict[str, Any]:
    text = _safe_str(report)
    words = text.split()
    sections_found = {k: bool(p.search(text)) for k, p in _SECTION_PATTERNS.items()}
    required = ("summary", "analysis", "recommendation")
    present = sum(1 for k in required if sections_found.get(k))
    sentences = _split_sentences(text)
    avg_sent_len = (
        round(sum(len(s.split()) for s in sentences) / len(sentences), 1) if sentences else 0.0
    )
    # Простий readability: коротші речення → вищий score (норма ~12–20 слів)
    readability = (
        round(max(0.0, min(1.0, 1.0 - abs(avg_sent_len - 16) / 24)), 3) if avg_sent_len else None
    )
    action_hits = sum(1 for kw in _ACTIONABILITY_KW if kw in text.lower())
    actionability = round(min(1.0, action_hits / 3.0), 3) if text else 0.0

    return {
        "report_char_count": len(text),
        "report_word_count": len(words),
        "sections_found": sections_found,
        "report_completeness_score": round(present / len(required), 3),
        "readability_score": readability,
        "avg_sentence_words": avg_sent_len,
        "actionability_score": actionability,
    }


def compute_classification_metrics(state: dict, report: str, task: dict | None) -> dict[str, Any]:
    findings = state.get("triage_findings") or trajectory.get("triage_findings") or {}
    dup = findings.get("duplicate") or {}
    cls = findings.get("classification") or {}

    pred_type = ""
    confidence_raw = None
    if isinstance(cls, dict) and cls.get("type"):
        pred_type = str(cls.get("type") or "")
        confidence_raw = cls.get("confidence")
    elif isinstance(dup, dict):
        confidence_raw = dup.get("confidence")
        if dup.get("is_duplicate"):
            pred_type = "duplicate"

    conf_score = None
    if isinstance(confidence_raw, str):
        conf_score = _CONFIDENCE_MAP.get(confidence_raw.strip().lower())

    out: dict[str, Any] = {
        "predicted_type": pred_type or None,
        "classification_confidence": confidence_raw,
        "classification_confidence_score": conf_score,
    }

    crit = (task or {}).get("success_criteria") or {}
    gold = crit.get("type_correct")
    if gold:
        if not isinstance(gold, (list, tuple)):
            gold = [gold]
        gold_l = [str(g).lower() for g in gold]
        pred_l = pred_type.lower()
        report_l = report.lower()
        type_match_structured = pred_l in gold_l if pred_l else False
        type_match_report = any(g in report_l for g in gold_l)
        out["gold_types"] = gold_l
        out["type_match_structured"] = type_match_structured
        out["type_match_report"] = type_match_report
        out["type_correct_effective"] = type_match_structured or type_match_report

    return out


def compute_duplicate_search_metrics(state: dict, trajectory: dict) -> dict[str, Any]:
    tool_results = trajectory.get("tool_results") or state.get("tool_results") or []
    searches = [r for r in tool_results if r.get("tool") == "search_similar_issues"]
    queries: list[str] = []
    for s in searches:
        args = s.get("args") or {}
        q = args.get("query") if isinstance(args, dict) else None
        if q:
            queries.append(str(q))

    similar = state.get("similar_issues") or []
    dup_findings = (state.get("triage_findings") or {}).get("duplicate") or {}
    dup_of = dup_findings.get("duplicate_of") if isinstance(dup_findings, dict) else None

    return {
        "search_query_count": len(searches),
        "unique_search_queries": len(set(queries)),
        "search_queries": queries[:10],
        "similar_issues_found": len(similar),
        "duplicate_of_predicted": dup_of,
        "search_tool_called": len(searches) > 0,
    }


def compute_tool_efficiency(state: dict, trajectory: dict) -> dict[str, Any]:
    tool_results = trajectory.get("tool_results") or state.get("tool_results") or []
    if not tool_results:
        return {
            "tool_calls_total": 0,
            "tool_success_rate": None,
            "tool_redundancy_rate": None,
            "tools_by_name": {},
        }

    sigs = [_tool_call_signature(r) for r in tool_results]
    unique_sigs = set(sigs)
    errors = sum(1 for r in tool_results if _tool_error(r))
    by_name: dict[str, int] = {}
    for r in tool_results:
        name = r.get("tool") or "?"
        by_name[name] = by_name.get(name, 0) + 1

    n = len(tool_results)
    return {
        "tool_calls_total": n,
        "tool_success_rate": round((n - errors) / n, 3),
        "tool_redundancy_rate": round(1.0 - len(unique_sigs) / n, 3) if n else None,
        "tool_error_count": errors,
        "tools_by_name": by_name,
        "tokens_per_tool_call": None,  # заповнюється в compute_extended_metrics
    }


def compute_stability_metrics(state: dict, trajectory: dict, report: str) -> dict[str, Any]:
    report_l = report.lower()
    budget_note = "budget limit" in report_l or "budget exceeded" in report_l
    return {
        "loop_count": int(state.get("loop_count") or trajectory.get("loop_count") or 0),
        "error_count": int(state.get("error_count") or trajectory.get("error_count") or 0),
        "retry_like_loop": int(state.get("loop_count") or 0) > 0,
        "budget_hit": budget_note,
        "has_report": bool(report.strip()),
        "error_recovery": bool(report.strip()) and int(state.get("error_count") or 0) > 0,
    }


def compute_hallucination_metrics(trajectory: dict, state: dict) -> dict[str, Any]:
    fc = trajectory.get("fact_check") or state.get("fact_check") or {}
    if not isinstance(fc, dict):
        fc = {}
    invented = fc.get("invented") or []
    total = int(fc.get("total_facts") or 0)
    grounded = int(fc.get("grounded_facts") or 0)
    gp = trajectory.get("grounding_passed")
    if gp is None:
        gp = state.get("grounding_passed")

    return {
        "grounding_passed": gp,
        "hallucination_detected": len(invented) > 0 or gp is False,
        "invented_fact_count": len(invented),
        "invented_per_report": len(invented),
        "total_facts_checked": total,
        "facts_grounded_rate": fc.get("facts_grounded_rate"),
        "fact_check_passed": fc.get("passed"),
    }


def compute_cost_latency(trajectory: dict, state: dict) -> dict[str, Any]:
    tok = trajectory.get("token_usage_est") or {}
    dur = trajectory.get("duration_s")
    if dur is None:
        dur = state.get("duration_s")
    ti = int(tok.get("input_tokens_est") or 0)
    to = int(tok.get("output_tokens_est") or 0)
    tc = int(trajectory.get("tool_calls") or state.get("tool_calls_count") or 0)
    cost = _estimate_cost_usd(ti, to) if (ti or to) else None

    return {
        "duration_s": dur,
        "token_input_est": ti,
        "token_output_est": to,
        "token_total_est": ti + to,
        "tokens_per_tool_call": round((ti + to) / tc, 1) if tc else None,
        "cost_per_task_usd_est": cost,
    }


def compute_rubric_hints(report: str, state: dict, trajectory: dict, task: dict | None) -> dict[str, Any]:
    """Критерії з tasks.json, які можна перевірити без повного score_result."""
    if not task:
        return {"task_id": None, "checks": {}}

    crit = task.get("success_criteria") or {}
    text = report.lower()
    tool_results = trajectory.get("tool_results") or state.get("tool_results") or []
    searches = [r for r in tool_results if r.get("tool") == "search_similar_issues"]

    checks: dict[str, Any] = {}

    min_q = crit.get("min_search_queries")
    if min_q is not None:
        n = len(searches)
        checks["min_search_queries"] = {
            "required": int(min_q),
            "actual": n,
            "passed": n >= int(min_q),
        }

    if crit.get("must_flag_security"):
        checks["must_flag_security"] = {
            "passed": any(k in text for k in _SECURITY_KW),
            "keywords_checked": list(_SECURITY_KW[:6]),
        }

    if crit.get("must_compare_root_causes"):
        checks["must_compare_root_causes"] = {
            "passed": any(k in text for k in _ROOT_CAUSE_KW),
        }

    return {"task_id": task.get("task_id"), "checks": checks}


def compute_extended_metrics(
    *,
    report: str,
    state: dict | None = None,
    trajectory: dict | None = None,
    task: dict | None = None,
) -> dict[str, Any]:
    """
    Збирає всі розширені метрики для одного прогону.
    state — фінальний стан графа; trajectory — експорт з main.py.
    task — опційно запис з evaluation/tasks.json (для gold type, rubric hints).
    """
    state = state or {}
    trajectory = trajectory or {}

    rq = compute_report_quality(report)
    clf = compute_classification_metrics(state, report, task)
    dup = compute_duplicate_search_metrics(state, trajectory)
    tools = compute_tool_efficiency(state, trajectory)
    stab = compute_stability_metrics(state, trajectory, report)
    hall = compute_hallucination_metrics(trajectory, state)
    cost = compute_cost_latency(trajectory, state)
    rubric = compute_rubric_hints(report, state, trajectory, task)

    tc = tools.get("tool_calls_total") or 0
    tot = cost.get("token_total_est") or 0
    if tc and tot:
        tools["tokens_per_tool_call"] = round(tot / tc, 1)

    return {
        "report_quality": rq,
        "classification": clf,
        "duplicate_search": dup,
        "tool_efficiency": tools,
        "stability": stab,
        "hallucination": hall,
        "cost_latency": cost,
        "rubric_autocheck": rubric,
    }


def aggregate_extended_metrics_summary(results: list[dict]) -> dict[str, Any]:
    """Середні / частки по eval-прогону з results[].extended_metrics."""
    n = 0
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    hall_detected = 0
    type_ok = 0
    type_total = 0
    search_q_sum = 0

    for r in results:
        if r.get("error"):
            continue
        em = r.get("extended_metrics")
        if not isinstance(em, dict):
            continue
        n += 1
        rq = em.get("report_quality") or {}
        for key in ("report_completeness_score", "actionability_score", "readability_score"):
            v = rq.get(key)
            if isinstance(v, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(v)
                counts[key] = counts.get(key, 0) + 1

        hall = em.get("hallucination") or {}
        if hall.get("hallucination_detected"):
            hall_detected += 1

        clf = em.get("classification") or {}
        if clf.get("gold_types") is not None:
            type_total += 1
            if clf.get("type_correct_effective"):
                type_ok += 1

        dup = em.get("duplicate_search") or {}
        search_q_sum += int(dup.get("search_query_count") or 0)

    if not n:
        return {"tasks_with_extended_metrics": 0}

    def _avg(key: str) -> float | None:
        c = counts.get(key, 0)
        return round(sums[key] / c, 3) if c else None

    return {
        "tasks_with_extended_metrics": n,
        "avg_report_completeness": _avg("report_completeness_score"),
        "avg_actionability": _avg("actionability_score"),
        "avg_readability": _avg("readability_score"),
        "hallucination_rate": round(hall_detected / n, 3),
        "avg_search_query_count": round(search_q_sum / n, 2),
        "type_correct_rate": round(type_ok / type_total, 3) if type_total else None,
        "type_correct_evaluated": type_total,
    }


def print_extended_metrics(metrics: dict[str, Any], *, issue_url: str = "") -> None:
    """Людський вивід розширених метрик у консоль (один issue)."""
    w = 62
    print("\n" + "═" * w)
    print("  РОЗШИРЕНІ МЕТРИКИ (один прогон)")
    if issue_url:
        print(f"  Issue: {issue_url}")
    print("═" * w)

    rq = metrics.get("report_quality") or {}
    print("\n  ■ Якість звіту")
    print(f"    Слів: {rq.get('report_word_count')}  |  Символів: {rq.get('report_char_count')}")
    print(f"    Повнота секцій (Summary/Analysis/Recommendation): {rq.get('report_completeness_score')}")
    print(f"    Секції: {rq.get('sections_found')}")
    print(f"    Readability (евристика): {rq.get('readability_score')}  |  Середня довжина речення (слів): {rq.get('avg_sentence_words')}")
    print(f"    Actionability (ключові слова): {rq.get('actionability_score')}")

    clf = metrics.get("classification") or {}
    print("\n  ■ Класифікація")
    print(f"    Тип: {clf.get('predicted_type')}  |  Confidence: {clf.get('classification_confidence')} (score={clf.get('classification_confidence_score')})")
    if clf.get("gold_types") is not None:
        print(f"    Еталон (eval): {clf.get('gold_types')}  |  match: structured={clf.get('type_match_structured')} report={clf.get('type_match_report')}")

    dup = metrics.get("duplicate_search") or {}
    print("\n  ■ Пошук дублікатів")
    print(f"    search_similar_issues викликів: {dup.get('search_query_count')}  |  унікальних запитів: {dup.get('unique_search_queries')}")
    if dup.get("search_queries"):
        print(f"    Запити: {dup.get('search_queries')}")
    print(f"    Знайдено схожих issues: {dup.get('similar_issues_found')}  |  duplicate_of: {dup.get('duplicate_of_predicted')}")

    tools = metrics.get("tool_efficiency") or {}
    print("\n  ■ Інструменти")
    print(f"    Всього викликів: {tools.get('tool_calls_total')}  |  success rate: {tools.get('tool_success_rate')}")
    print(f"    Redundancy (повторні сигнатури): {tools.get('tool_redundancy_rate')}")
    print(f"    По іменах: {tools.get('tools_by_name')}")

    stab = metrics.get("stability") or {}
    print("\n  ■ Стабільність")
    print(f"    loop_count: {stab.get('loop_count')}  |  error_count: {stab.get('error_count')}  |  budget_hit: {stab.get('budget_hit')}")

    hall = metrics.get("hallucination") or {}
    print("\n  ■ Галюцинації / grounding")
    print(f"    grounding_passed: {hall.get('grounding_passed')}  |  invented facts: {hall.get('invented_fact_count')}")
    print(f"    facts_grounded_rate: {hall.get('facts_grounded_rate')}  |  fact_check_passed: {hall.get('fact_check_passed')}")

    cost = metrics.get("cost_latency") or {}
    print("\n  ■ Час і вартість (оцінка)")
    print(
        f"    duration_s: {cost.get('duration_s')}  |  tokens in/out/total: "
        f"{cost.get('token_input_est')}/{cost.get('token_output_est')}/{cost.get('token_total_est')}"
    )
    print(f"    tokens_per_tool_call: {cost.get('tokens_per_tool_call')}  |  cost_usd_est: {cost.get('cost_per_task_usd_est')}")

    rub = metrics.get("rubric_autocheck") or {}
    checks = rub.get("checks") or {}
    if checks:
        print(f"\n  ■ Автоперевірка критеріїв з tasks.json (task={rub.get('task_id')})")
        for name, detail in checks.items():
            print(f"    • {name}: {detail}")

    print("\n" + "═" * w + "\n")
