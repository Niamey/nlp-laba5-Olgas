"""
IssueTriageState — повна схема стану LangGraph агента.
Розділена на durable (зберігається між запусками) і scratchpad (transient).
"""
from __future__ import annotations
from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages


class IssueTriageState(TypedDict):
    # ═══════════════════════════════════════════════
    # DURABLE STATE — зберігається чекпойнтером
    # ═══════════════════════════════════════════════

    # Вхідні дані
    issue_url: str
    task_hint: Optional[str]        # підказка від юзера: duplicate|code_area|stale|classify
    user_prompt: Optional[str]      # довільні інструкції користувача для цього прогону

    # Результати decompose
    repo_owner: Optional[str]
    repo_name: Optional[str]
    issue_number: Optional[int]

    # Результати planning
    task_type: Optional[str]        # duplicate | code_area | stale | classify
    plan: List[str]                 # кроки плану

    # Дані з GitHub API
    fetched_issue: Optional[dict]   # повний JSON issue
    similar_issues: List[dict]      # знайдені схожі issues
    repo_structure: Optional[str]   # дерево файлів (для code_area)

    # Accumulated evidence
    tool_results: List[dict]        # всі виклики інструментів з результатами
    triage_findings: dict           # структуровані знахідки по категоріях

    # Outputs
    triage_report: Optional[str]    # фінальний звіт
    grounding_passed: bool          # чи пройшла валідація
    fact_check: Optional[dict]      # regex fact-check: invented #N/@user/file paths + facts_grounded_rate

    # Control flow
    needs_more_info: bool
    should_escalate: bool
    tool_calls_count: int           # budget tracking
    loop_count: int                 # скільки разів зробили loop
    run_id: str

    # ═══════════════════════════════════════════════
    # SCRATCHPAD — transient, не архівується
    # ═══════════════════════════════════════════════

    messages: Annotated[List[Any], add_messages]
    error_count: int
    last_error: Optional[str]
    human_feedback: Optional[str]   # заповнюється при HIL interrupt


def initial_state(
    issue_url: str,
    task_hint: Optional[str] = None,
    user_prompt: Optional[str] = None,
    run_id: str = "",
) -> IssueTriageState:
    """Створює початковий стан для нового запуску."""
    import uuid
    return IssueTriageState(
        issue_url=issue_url,
        task_hint=task_hint,
        user_prompt=(user_prompt or "").strip() or None,
        repo_owner=None,
        repo_name=None,
        issue_number=None,
        task_type=None,
        plan=[],
        fetched_issue=None,
        similar_issues=[],
        repo_structure=None,
        tool_results=[],
        triage_findings={},
        triage_report=None,
        grounding_passed=False,
        fact_check=None,
        needs_more_info=False,
        should_escalate=False,
        tool_calls_count=0,
        loop_count=0,
        run_id=run_id or str(uuid.uuid4()),
        messages=[],
        error_count=0,
        last_error=None,
        human_feedback=None,
    )
