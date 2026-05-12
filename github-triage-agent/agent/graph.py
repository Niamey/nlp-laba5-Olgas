"""
graph.py — збирає LangGraph StateGraph.
Включає: conditional edges, MemorySaver checkpointer, HIL interrupt.
"""
from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import IssueTriageState
from agent.nodes.decompose  import decompose_task
from agent.nodes.fetch       import fetch_issue_node
from agent.nodes.planner     import plan_node
from agent.nodes.workers     import (
    duplicate_point, code_area_point, stale_point, classify_point
)
from agent.nodes.completion  import draft_answer, grounding_validator, completion_check, human_review
from agent.nodes.router      import (
    route_after_decompose, route_by_task_type, route_after_completion
)

logger = logging.getLogger(__name__)


# ─── Error sink node ────────────────────────────────
async def error_sink(state: IssueTriageState) -> dict:
    """Термінальний вузол для помилок — форматує error report."""
    error = state.get("last_error", "Unknown error")
    return {
        "triage_report": (
            f"# Triage Failed\n\n"
            f"**Issue:** {state['issue_url']}\n"
            f"**Error:** {error}\n"
            f"**Tool calls used:** {state['tool_calls_count']}\n"
            f"**Error count:** {state['error_count']}\n\n"
            f"Please check the issue URL and try again."
        ),
        "grounding_passed": False,
    }


# ─── Budget exceeded sink ────────────────────────────
async def budget_sink(state: IssueTriageState) -> dict:
    """Досягнуто ліміт tool calls — повертаємо частковий звіт."""
    partial = state.get("triage_report") or "Partial analysis — budget exceeded"
    return {
        "triage_report": (
            f"{partial}\n\n"
            f"---\n"
            f"*Note: Analysis stopped after {state['tool_calls_count']} tool calls (budget limit).*"
        ),
    }


def build_graph(checkpointer=None) -> StateGraph:
    """
    Будує та компілює граф.
    checkpointer=None → MemorySaver за замовчуванням.
    """
    g = StateGraph(IssueTriageState)

    # ── Nodes ──────────────────────────────────────
    g.add_node("decompose_task",      decompose_task)
    g.add_node("fetch_issue",         fetch_issue_node)
    g.add_node("plan_node",           plan_node)
    g.add_node("route_use_selector",  _passthrough)   # router — лише conditional edges
    g.add_node("duplicate_point",     duplicate_point)
    g.add_node("code_area_point",     code_area_point)
    g.add_node("stale_point",         stale_point)
    g.add_node("classify_point",      classify_point)
    g.add_node("draft_answer",        draft_answer)
    g.add_node("grounding_validator", grounding_validator)
    g.add_node("completion_check",    completion_check)
    g.add_node("human_review",        human_review)
    g.add_node("error_sink",          error_sink)
    g.add_node("budget_sink",         budget_sink)

    # ── Entry ──────────────────────────────────────
    g.set_entry_point("decompose_task")

    # ── decompose → (validate URL) → fetch ─────────
    g.add_conditional_edges(
        "decompose_task",
        route_after_decompose,
        {
            "fetch_issue":    "fetch_issue",
            "end_with_error": "error_sink",
        }
    )

    # fetch → plan (завжди — plan знає що робити навіть з помилкою)
    g.add_edge("fetch_issue", "plan_node")

    # plan → route
    g.add_edge("plan_node", "route_use_selector")

    # ── Conditional routing за task_type ───────────
    g.add_conditional_edges(
        "route_use_selector",
        route_by_task_type,
        {
            "duplicate_point":  "duplicate_point",
            "code_area_point":  "code_area_point",
            "stale_point":      "stale_point",
            "classify_point":   "classify_point",
            "budget_exceeded":  "budget_sink",
            "not_found":        "error_sink",
        }
    )

    # ── Всі воркери → draft ─────────────────────────
    for worker in ["duplicate_point", "code_area_point", "stale_point", "classify_point"]:
        g.add_edge(worker, "draft_answer")

    # draft → validate → check
    g.add_edge("draft_answer",        "grounding_validator")
    g.add_edge("grounding_validator", "completion_check")

    # ── completion_check → 3 шляхи ─────────────────
    g.add_conditional_edges(
        "completion_check",
        route_after_completion,
        {
            "human_review":       "human_review",
            "route_use_selector": "route_use_selector",   # loop
            "end_success":        END,
            "force_end":          "budget_sink",
            "end_with_error":     "error_sink",
        }
    )

    # human_review → draft (ревізія) або END
    g.add_edge("human_review", "draft_answer")

    # Sinks → END
    g.add_edge("error_sink",  END)
    g.add_edge("budget_sink", END)

    # ── Compile ─────────────────────────────────────
    cp = checkpointer or MemorySaver()
    compiled = g.compile(
        checkpointer=cp,
        interrupt_before=["human_review"],   # ← HIL interrupt для demo
    )
    logger.info("Graph compiled with %d nodes", len(g.nodes))
    return compiled


async def _passthrough(state: IssueTriageState) -> dict:
    """Router node — не змінює стан, просто conditional edges."""
    return {}
