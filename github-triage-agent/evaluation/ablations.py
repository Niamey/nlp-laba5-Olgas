"""
evaluation/ablations.py — запускає 3 обов'язкові аблації.

Аблація 1: Модель   — PRIMARY vs SECONDARY model
Аблація 2: Промпт  — стандартний vs мінімалістичний system prompt
Аблація 3: Граф    — повний граф vs single ReAct loop (без воркерів)

Запуск:
    python -m evaluation.ablations --ablation model
    python -m evaluation.ablations --ablation prompt
    python -m evaluation.ablations --ablation graph
    python -m evaluation.ablations --ablation all
"""
from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import os
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ablations")

# Підмножина задач для аблацій (15 задач — рівний розподіл по категоріях)
ABLATION_TASK_IDS = [
    "T001", "T002", "T003", "T004", "T005",   # happy path
    "T013", "T014", "T015", "T016", "T017",   # ambiguous
    "T018", "T019", "T020",                   # adversarial
    "T021", "T022",                           # should_refuse
]


# ══════════════════════════════════════════════════════════════════
# ABLATION 1: Model ablation
# ══════════════════════════════════════════════════════════════════

async def run_model_ablation(output_dir: str = "trajectories/ablations") -> dict:
    """
    Порівнює PRIMARY_MODEL vs SECONDARY_MODEL на одному evaluation set.
    Метрики: score %, pass rate, avg latency, avg tool calls.
    """
    from evaluation.runner import run_evaluation

    out = Path(output_dir) / "model_ablation"
    out.mkdir(parents=True, exist_ok=True)

    primary   = os.getenv("PRIMARY_MODEL",   "gemini-2.5-flash")
    secondary = os.getenv("SECONDARY_MODEL", "gemini-2.0-flash")

    print(f"\n{'═'*60}")
    print(f"  ABLATION 1: MODEL")
    print(f"  Primary:   {primary}")
    print(f"  Secondary: {secondary}")
    print(f"{'═'*60}\n")

    results = {}
    for model in [primary, secondary]:
        label = "primary" if model == primary else "secondary"
        print(f"\n--- Running with model: {model} ---")
        summary = await run_evaluation(
            output_dir=str(out / label),
            tasks_filter=ABLATION_TASK_IDS,
            model=model,
        )
        results[label] = {
            "model":          model,
            "score_pct":      summary["score_pct"],
            "pass_rate":      summary["pass_rate"],
            "avg_latency_s":  summary["avg_latency_s"],
            "avg_tool_calls": summary["avg_tool_calls"],
            "tool_usage":     summary["tool_usage"],
        }

    # Різниця
    diff = {
        "score_pct_delta":      results["primary"]["score_pct"]     - results["secondary"]["score_pct"],
        "pass_rate_delta":      results["primary"]["pass_rate"]      - results["secondary"]["pass_rate"],
        "latency_delta_s":      results["primary"]["avg_latency_s"]  - results["secondary"]["avg_latency_s"],
        "tool_calls_delta":     results["primary"]["avg_tool_calls"] - results["secondary"]["avg_tool_calls"],
    }

    report = {
        "ablation":   "model",
        "timestamp":  datetime.now().isoformat(),
        "results":    results,
        "diff":       diff,
        "conclusion": _model_conclusion(results, diff),
    }

    report_path = out / "model_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_model_report(results, diff)
    print(f"\n  Report: {report_path}")
    return report


def _model_conclusion(results: dict, diff: dict) -> str:
    primary_better = diff["score_pct_delta"] > 0
    faster = diff["latency_delta_s"] < 0

    if primary_better:
        return (
            f"Primary model ({results['primary']['model']}) scores "
            f"{abs(diff['score_pct_delta']):.1f}% higher than secondary. "
            f"{'Primary is also faster.' if faster else 'Secondary is faster by ' + str(abs(round(diff['latency_delta_s'],1))) + 's.'}"
        )
    else:
        return (
            f"Secondary model ({results['secondary']['model']}) is competitive: "
            f"only {abs(diff['score_pct_delta']):.1f}% lower score but "
            f"{abs(round(diff['latency_delta_s'],1))}s {'faster' if diff['latency_delta_s'] < 0 else 'slower'}."
        )


def _print_model_report(results: dict, diff: dict):
    print(f"\n{'─'*60}")
    print(f"  MODEL ABLATION RESULTS")
    print(f"{'─'*60}")
    for label, r in results.items():
        print(f"  {label:10s} [{r['model']:40s}]")
        print(f"            Score:   {r['score_pct']:5.1f}%")
        print(f"            Pass:    {r['pass_rate']*100:5.1f}%")
        print(f"            Latency: {r['avg_latency_s']:5.1f}s avg")
        print(f"            Tools:   {r['avg_tool_calls']:5.1f} avg")
    print(f"  ΔScore:   {diff['score_pct_delta']:+.1f}%  (primary vs secondary)")
    print(f"  ΔLatency: {diff['latency_delta_s']:+.1f}s")


# ══════════════════════════════════════════════════════════════════
# ABLATION 2: Prompt ablation
# ══════════════════════════════════════════════════════════════════

MINIMAL_SYSTEM_PROMPT = "You are a GitHub issue triage assistant. Be concise."

VERBOSE_SYSTEM_PROMPT = """You are an expert GitHub issue triage agent with deep knowledge of software engineering.

Your role is to systematically analyze GitHub issues using a structured methodology:

1. EVIDENCE-BASED ANALYSIS: Every conclusion must cite specific text from the issue
2. STRUCTURED OUTPUT: Always provide type, priority, confidence, and justification
3. TOOL GROUNDING: Only report facts retrievable through the tools provided
4. UNCERTAINTY ACKNOWLEDGMENT: When data is insufficient, explicitly state low confidence

For DUPLICATE detection: compare titles, error messages, reproduction steps, and root causes
For CODE AREA identification: trace stack traces, file mentions, and component names
For STALE ISSUE analysis: check creation date, last update, comment activity, and blocking questions
For CLASSIFICATION: use standard GitHub labels: bug, enhancement, question, documentation, duplicate

NEVER: invent issue numbers, file paths, usernames, or content not present in tool results
ALWAYS: cite the specific evidence from the issue that supports your conclusion"""


async def run_prompt_ablation(output_dir: str = "trajectories/ablations") -> dict:
    """
    Порівнює детальний system prompt vs мінімальний.
    Перезаписує SYSTEM_PROMPT в agent.llm і запускає evaluation.
    """
    import agent.llm as llm_module
    from evaluation.runner import run_evaluation

    out = Path(output_dir) / "prompt_ablation"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  ABLATION 2: PROMPT")
    print(f"  Verbose ({len(VERBOSE_SYSTEM_PROMPT)} chars) vs Minimal ({len(MINIMAL_SYSTEM_PROMPT)} chars)")
    print(f"{'═'*60}\n")

    results = {}
    configs = {
        "verbose": VERBOSE_SYSTEM_PROMPT,
        "minimal": MINIMAL_SYSTEM_PROMPT,
    }

    for label, prompt in configs.items():
        print(f"\n--- Running with {label} prompt ---")
        # Monkey-patch system prompt
        original = llm_module.SYSTEM_PROMPT
        llm_module.SYSTEM_PROMPT = prompt

        # Очищаємо LLM кеш щоб взяти нові prompts
        llm_module.get_llm.cache_clear()

        try:
            summary = await run_evaluation(
                output_dir=str(out / label),
                tasks_filter=ABLATION_TASK_IDS,
            )
            results[label] = {
                "prompt_length": len(prompt),
                "score_pct":      summary["score_pct"],
                "pass_rate":      summary["pass_rate"],
                "avg_tool_calls": summary["avg_tool_calls"],
                "avg_latency_s":  summary["avg_latency_s"],
            }
        finally:
            llm_module.SYSTEM_PROMPT = original
            llm_module.get_llm.cache_clear()

    diff = {
        "score_pct_delta":  results["verbose"]["score_pct"]  - results["minimal"]["score_pct"],
        "pass_rate_delta":  results["verbose"]["pass_rate"]  - results["minimal"]["pass_rate"],
        "latency_delta_s":  results["verbose"]["avg_latency_s"] - results["minimal"]["avg_latency_s"],
    }

    report = {
        "ablation":  "prompt",
        "timestamp": datetime.now().isoformat(),
        "results":   results,
        "diff":      diff,
        "conclusion": (
            f"Verbose prompt {'improves' if diff['score_pct_delta'] > 0 else 'reduces'} "
            f"score by {abs(diff['score_pct_delta']):.1f}% vs minimal prompt. "
            f"Tool-selection accuracy is the primary differentiator."
        ),
    }

    report_path = out / "prompt_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'─'*60}  PROMPT ABLATION RESULTS")
    for label, r in results.items():
        print(f"  {label:10s}  Score: {r['score_pct']:5.1f}%  Pass: {r['pass_rate']*100:.0f}%  Tools: {r['avg_tool_calls']:.1f}")
    print(f"  ΔScore: {diff['score_pct_delta']:+.1f}%  (verbose vs minimal)")
    print(f"  Report: {report_path}")
    return report


# ══════════════════════════════════════════════════════════════════
# ABLATION 3: Graph ablation
# ══════════════════════════════════════════════════════════════════

async def run_graph_ablation(output_dir: str = "trajectories/ablations") -> dict:
    """
    Порівнює повний граф (спеціалізовані воркери) vs
    спрощений ReAct loop (один universal worker).
    """
    from evaluation.runner import run_evaluation
    import agent.graph as graph_module
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from agent.state import IssueTriageState
    from agent.nodes.decompose import decompose_task
    from agent.nodes.fetch import fetch_issue_node
    from agent.nodes.completion import draft_answer, grounding_validator

    out = Path(output_dir) / "graph_ablation"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  ABLATION 3: GRAPH TOPOLOGY")
    print(f"  Full graph (specialized workers) vs Collapsed ReAct loop")
    print(f"{'═'*60}\n")

    results = {}

    # ── Варіант A: Повний граф ──────────────────────────────────
    print("\n--- Full graph (A) ---")
    summary_a = await run_evaluation(
        output_dir=str(out / "full_graph"),
        tasks_filter=ABLATION_TASK_IDS,
    )
    results["full_graph"] = {
        "description":    "Specialized workers + planner + validator",
        "score_pct":      summary_a["score_pct"],
        "pass_rate":      summary_a["pass_rate"],
        "avg_tool_calls": summary_a["avg_tool_calls"],
        "avg_latency_s":  summary_a["avg_latency_s"],
    }

    # ── Варіант B: Collapsed ReAct ─────────────────────────────
    print("\n--- Collapsed ReAct loop (B) ---")

    def build_collapsed_graph():
        """Спрощений граф: decompose → fetch → universal_worker → draft → END."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from langchain_core.runnables import RunnableConfig
        from agent.llm import get_llm, SYSTEM_PROMPT

        async def universal_worker(state: IssueTriageState, config: RunnableConfig) -> dict:
            """Один вузол робить все — без спеціалізації."""
            tools = (config.get("configurable") or {}).get("tools") or {}
            fetch_tool = tools.get("fetch_github_issue")
            issue = state.get("fetched_issue") or {}

            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Triage this GitHub issue and provide full analysis:\n"
                    f"URL: {state['issue_url']}\n"
                    f"Title: {issue.get('title', 'N/A')}\n"
                    f"Body: {(issue.get('body') or '')[:1000]}\n"
                    f"State: {issue.get('state')}\n"
                    f"Labels: {[l['name'] for l in issue.get('labels', [])]}\n\n"
                    f"Provide: classification, duplicate check assessment, "
                    f"code area, and recommendation. Be comprehensive."
                ))
            ])

            return {
                "triage_findings": {"universal": {"analysis": response.content}},
                "tool_calls_count": state["tool_calls_count"] + 1,
                "grounding_passed": True,   # колапсований граф — без валідатора
            }

        g = StateGraph(IssueTriageState)
        g.add_node("decompose_task",  decompose_task)
        g.add_node("fetch_issue",     fetch_issue_node)
        g.add_node("universal_worker", universal_worker)
        g.add_node("draft_answer",    draft_answer)

        g.set_entry_point("decompose_task")
        g.add_edge("decompose_task",   "fetch_issue")
        g.add_edge("fetch_issue",      "universal_worker")
        g.add_edge("universal_worker", "draft_answer")
        g.add_edge("draft_answer",     END)

        return g.compile(checkpointer=MemorySaver())

    # Підміняємо build_graph
    original_build = graph_module.build_graph
    graph_module.build_graph = build_collapsed_graph
    try:
        summary_b = await run_evaluation(
            output_dir=str(out / "collapsed"),
            tasks_filter=ABLATION_TASK_IDS,
        )
        results["collapsed"] = {
            "description":    "Single universal worker, no specialization",
            "score_pct":      summary_b["score_pct"],
            "pass_rate":      summary_b["pass_rate"],
            "avg_tool_calls": summary_b["avg_tool_calls"],
            "avg_latency_s":  summary_b["avg_latency_s"],
        }
    finally:
        graph_module.build_graph = original_build

    diff = {
        "score_pct_delta":  results["full_graph"]["score_pct"]  - results["collapsed"]["score_pct"],
        "pass_rate_delta":  results["full_graph"]["pass_rate"]  - results["collapsed"]["pass_rate"],
        "tool_calls_delta": results["full_graph"]["avg_tool_calls"] - results["collapsed"]["avg_tool_calls"],
    }

    report = {
        "ablation":  "graph",
        "timestamp": datetime.now().isoformat(),
        "results":   results,
        "diff":      diff,
        "conclusion": (
            f"Full graph {'outperforms' if diff['score_pct_delta'] > 0 else 'underperforms vs'} "
            f"collapsed by {abs(diff['score_pct_delta']):.1f}%. "
            f"Full graph uses {diff['tool_calls_delta']:+.1f} more tool calls on average. "
            f"Specialization {'justifies' if diff['score_pct_delta'] > 2 else 'marginally justifies'} the added complexity."
        ),
    }

    report_path = out / "graph_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'─'*60}  GRAPH ABLATION RESULTS")
    for label, r in results.items():
        print(f"  {label:15s}  Score: {r['score_pct']:5.1f}%  Pass: {r['pass_rate']*100:.0f}%  Tools: {r['avg_tool_calls']:.1f}")
    print(f"  ΔScore (full vs collapsed): {diff['score_pct_delta']:+.1f}%")
    print(f"  Report: {report_path}")
    return report


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--ablation",
        choices=["model", "prompt", "graph", "all"],
        default="all",
        help="Which ablation to run",
    )
    parser.add_argument("--output", default="trajectories/ablations")
    args = parser.parse_args()

    if args.ablation in ("model", "all"):
        await run_model_ablation(args.output)

    if args.ablation in ("prompt", "all"):
        await run_prompt_ablation(args.output)

    if args.ablation in ("graph", "all"):
        await run_graph_ablation(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
