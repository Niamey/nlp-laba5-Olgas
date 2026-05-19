"""
decompose_task — перший вузол графа.
Парсить URL issue, витягує repo/number, визначає базовий task_hint.
"""
from __future__ import annotations
import re
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.llm_messages import aimessage_from_llm
from agent.user_instructions import normalize_task_hint, user_instructions_block
from agent.state import IssueTriageState
from agent.llm import get_fast_llm, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)"
)

_VALID_HINTS = frozenset({"duplicate", "code_area", "stale", "classify"})


async def decompose_task(state: IssueTriageState) -> dict:
    """
    Парсить URL, витягує repo owner/name/issue number.
    Визначає task_type з task_hint якщо є.
    """
    url = state["issue_url"].strip()
    match = GITHUB_URL_RE.match(url)

    # --- Валідація URL ---
    if not match:
        logger.warning("Invalid GitHub issue URL: %s", url)
        return {
            "last_error": f"Invalid GitHub issue URL: {url}. Expected format: https://github.com/owner/repo/issues/N",
            "error_count": state["error_count"] + 1,
            "task_type": "invalid",
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                AIMessage(content=f"ERROR: Cannot parse URL: {url}")
            ],
        }

    owner, repo, number = match.group(1), match.group(2), int(match.group(3))
    logger.info("Decomposed: %s/%s#%d", owner, repo, number)

    # --- Визначення task_type з hint (якщо є) ---
    task_type = None
    hint = (state.get("task_hint") or "").strip()

    hint_llm_msg = None
    # Режим: --hint duplicate|… або перше слово в --prompt
    mode_from_prompt = normalize_task_hint(state.get("user_prompt"))
    if mode_from_prompt and not task_type:
        task_type = mode_from_prompt

    if hint:
        low = hint.lower()
        first = low.split()[0] if low else ""
        # CLI: --hint duplicate — без додаткового LLM (надійніше, ніж fast_llm)
        if low in _VALID_HINTS:
            task_type = low
        elif first in _VALID_HINTS:
            task_type = first
        else:
            llm = get_fast_llm()
            hint_msg = (
                f"The user wants to triage GitHub issue {url} with this hint: '{hint}'\n"
                "Choose ONE task type from: duplicate, code_area, stale, classify\n"
                "Respond with ONLY the task type word, nothing else."
            )
            hint_msg += user_instructions_block(state)
            resp = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=hint_msg),
            ])
            hint_llm_msg = resp
            candidate = resp.content.strip().lower().split()[0] if resp.content.strip() else ""
            if candidate in _VALID_HINTS:
                task_type = candidate

    parsed_body = (
        f"Parsed issue: {owner}/{repo}#{number}\n"
        f"Task type from hint: {task_type or 'to be determined after fetch'}"
    )
    parsed_ai = (
        aimessage_from_llm(parsed_body, hint_llm_msg)
        if hint_llm_msg is not None
        else AIMessage(content=parsed_body)
    )

    return {
        "repo_owner":   owner,
        "repo_name":    repo,
        "issue_number": number,
        "task_type":    task_type,   # може бути None — тоді plan_node вирішить
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            parsed_ai,
        ],
    }
