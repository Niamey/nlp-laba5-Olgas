"""
plan_node — читає завантажений issue і будує план тріажу.
Також визначає остаточний task_type якщо він ще не встановлений.
"""
from __future__ import annotations
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.llm_messages import aimessage_from_llm
from agent.user_instructions import user_instructions_block
from agent.prompt_quality import detect_intents
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


PLAN_PROMPT = """You are triaging a GitHub issue. Based on the issue content below (and any user instructions), determine:

1. TASK TYPE — choose exactly ONE:
   - "duplicate": the issue appears to be a duplicate; or user asks to find similar/duplicate issues
   - "code_area": we need to identify which code module/file is affected; or user asks where in the code
   - "stale": the issue is old (>6 months without activity); or user asks about staleness / what to do with old issues
   - "classify": general classification into bug/feature/question/documentation; or user asks for type/labels

If "User instructions" are present below, they are the strongest signal for picking task_type.
Map common Ukrainian/English keywords:
   - "дублікат", "схожі", "duplicate", "similar"           → duplicate
   - "модуль", "файл", "код", "where in code", "module"    → code_area
   - "застарілий", "старий", "stale", "old", "outdated"    → stale
   - "класиф", "тип", "bug", "feature", "label", "classify"→ classify

2. ACTION PLAN — 3-5 specific steps to triage this issue

Issue data:
{issue_text}

Respond with ONLY valid JSON in this exact format:
{{
  "task_type": "duplicate|code_area|stale|classify",
  "reasoning": "one sentence why (mention if user instructions drove the choice)",
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

    user_prompt_text = (state.get("user_prompt") or "").strip()
    detected = detect_intents(user_prompt_text) if user_prompt_text else []

    if state.get("task_type"):
        hint_line = f"\nNote: the user hinted this is a '{state['task_type']}' task — confirm or override."
        issue_text += hint_line
    else:
        if user_prompt_text:
            issue_text += (
                "\nNote: no explicit --hint was given. The user provided instructions below — "
                "use them as the PRIMARY signal to pick task_type."
            )
            if len(detected) > 1:
                issue_text += (
                    f"\nDetected MULTIPLE intents in user prompt: {', '.join(detected)}. "
                    "Pick the SINGLE strongest task_type and state in 'reasoning' that other intents "
                    "(e.g. 'also classify the type') will be addressed inside the report, "
                    "not as a separate triage branch."
                )
            elif len(detected) == 0:
                issue_text += (
                    "\nNo clear task_type keywords detected in the user prompt — "
                    "infer task_type from the issue content + general meaning of the prompt; "
                    "if still unclear, default to 'classify' and mention low confidence in 'reasoning'."
                )
        else:
            issue_text += "\nNote: no --hint and no user prompt — pick task_type purely from the issue content."
    issue_text += user_instructions_block(state, prefix="User instructions (PRIMARY signal for task_type when no --hint):")

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
                aimessage_from_llm(
                    (
                        f"Plan created:\n"
                        f"Task type: {task_type} ({reasoning})\n"
                        f"Steps: " + " → ".join(plan)
                    ),
                    response,
                )
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
