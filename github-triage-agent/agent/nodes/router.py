"""
router.py — функції маршрутизації для conditional edges.
Чисті функції без side effects.
"""
from __future__ import annotations
from agent.state import IssueTriageState

MAX_TOOL_CALLS = int(__import__("os").getenv("MAX_TOOL_CALLS", "25"))
MAX_LOOPS      = int(__import__("os").getenv("MAX_LOOPS", "3"))


def route_after_decompose(state: IssueTriageState) -> str:
    """Після decompose: якщо URL невалідний — одразу END."""
    if state.get("task_type") == "invalid" or state.get("error_count", 0) > 2:
        return "end_with_error"
    return "fetch_issue"


def route_by_task_type(state: IssueTriageState) -> str:
    """
    Головний роутер — розподіляє після plan_node.
    Перевіряє бюджет перед роутингом.
    """
    # Перевірка бюджету
    if state["tool_calls_count"] >= MAX_TOOL_CALLS:
        return "budget_exceeded"

    if state.get("task_type") == "not_found":
        return "not_found"

    task_type = state.get("task_type", "classify")
    routes = {
        "duplicate": "duplicate_point",
        "code_area": "code_area_point",
        "stale":     "stale_point",
        "classify":  "classify_point",
    }
    return routes.get(task_type, "classify_point")


def route_after_completion(state: IssueTriageState) -> str:
    """
    Після completion_check: вирішує що далі.
    Порядок пріоритетів: error > budget > escalate > more_info > done
    """
    if state["error_count"] > 3:
        return "end_with_error"

    if state["tool_calls_count"] >= MAX_TOOL_CALLS:
        return "force_end"

    if state["loop_count"] >= MAX_LOOPS:
        return "force_end"

    if state.get("should_escalate"):
        return "human_review"

    if state.get("needs_more_info"):
        return "route_use_selector"

    return "end_success"
