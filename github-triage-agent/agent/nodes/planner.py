"""
plan_node — читає завантажений issue і будує план тріажу.
Також визначає остаточний task_type якщо він ще не встановлений.
"""
from __future__ import annotations
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import IssueTriageState
from agent.llm import get_llm, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES = frozenset({"duplicate", "code_area", "stale", "classify"})


def _format_issue_for_llm(issue: dict) -> str:
    """Форматує issue dict у компактний текст для LLM."""
    labels = ", ".join(l["name"] for l in issue.get("labels", []))
    body = (issue.get("body") or "")[:2000]  # обрізаємо щоб не перевищити контекст
    return (
        f"Title: {issue.get('title', 'N/A')}\n"
        f"State: {issue.get('state', 'N/A')}\n"
        f"Labels: {labels or 'none'}\n"
        f"Author: {issue.get('user', {}).get('login', 'N/A')}\n"
        f"Created: {issue.get('created_at', 'N/A')}\n"
        f"Updated: {issue.get('updated_at', 'N/A')}\n"
        f"Comments: {issue.get('comments', 0)}\n"
        f"Body:\n{body}"
    )


PLAN_PROMPT = """You are triaging a GitHub issue. Based on the issue content below, determine:

1. TASK TYPE — choose exactly ONE:
   - "duplicate": the issue appears to be a duplicate of an existing issue
   - "code_area": we need to identify which code module/file is affected
   - "stale": the issue is old and needs staleness analysis (>6 months without activity)
   - "classify": general classification into bug/feature/question/documentation

2. ACTION PLAN — 3-5 specific steps to triage this issue

Issue data:
{issue_text}

Respond with ONLY valid JSON in this exact format:
{{
  "task_type": "duplicate|code_area|stale|classify",
  "reasoning": "one sentence why",
  "plan": ["step 1", "step 2", "step 3"]
}}"""


async def plan_node(state: IssueTriageState) -> dict:
    """
    Визначає task_type і будує план дій.
    Якщо task_type вже встановлений з decompose — підтверджує або коригує.
    """
    issue = state.get("fetched_issue")

    # Якщо issue не завантажений — fallback
    if not issue:
        return {
            "task_type": state.get("task_type") or "classify",
            "plan": ["Fetch issue data", "Classify based on available information"],
            "messages": [AIMessage(content="Warning: no issue data, defaulting to classify")],
        }

    issue_text = _format_issue_for_llm(issue)

    # Якщо task_type вже є — плануємо з урахуванням цього
    if state.get("task_type"):
        hint_line = f"\nNote: the user hinted this is a '{state['task_type']}' task — confirm or override."
        issue_text += hint_line

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PLAN_PROMPT.format(issue_text=issue_text)),
    ])

    # Парсимо JSON відповідь
    try:
        raw = response.content.strip()
        # Прибираємо можливі markdown backticks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        task_type = parsed.get("task_type", "classify")
        if task_type not in _VALID_TASK_TYPES:
            task_type = "classify"

        # Підказка з CLI/decompose не перезаписується планувальником (LLM часто обирає classify)
        locked = state.get("task_type")
        if locked in _VALID_TASK_TYPES:
            if task_type != locked:
                logger.info("Plan: overriding LLM task_type=%s → %s (user hint)", task_type, locked)
            task_type = locked

        plan = parsed.get("plan", ["Analyze issue", "Draft report"])
        reasoning = parsed.get("reasoning", "")

        logger.info("Plan: task_type=%s, steps=%d", task_type, len(plan))

        return {
            "task_type": task_type,
            "plan": plan,
            "messages": [
                AIMessage(content=(
                    f"Plan created:\n"
                    f"Task type: {task_type} ({reasoning})\n"
                    f"Steps: " + " → ".join(plan)
                ))
            ],
        }

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse plan response: %s", e)
        # Fallback до classify
        return {
            "task_type": state.get("task_type") or "classify",
            "plan": ["Classify the issue", "Check labels and description", "Draft report"],
            "messages": [AIMessage(content=f"Plan parse failed ({e}), using default classify flow")],
        }
