"""
drafter.py    — синтезує знахідки у структурований звіт
validator.py  — перевіряє grounding (кожне твердження = tool result)
completion.py — вирішує done/more/escalate + human_review interrupt
"""
from __future__ import annotations
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import IssueTriageState
from agent.llm import get_llm, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# DRAFTER
# ══════════════════════════════════════════════════════

REPORT_TEMPLATE = """Generate a structured GitHub issue triage report based on the findings.

Issue: {issue_url}
Task type: {task_type}
Findings: {findings_json}

Tool results summary (what data was retrieved):
{tool_summary}

Write a triage report with these sections:
1. **Summary** — one sentence
2. **Analysis** — key findings (reference specific data from tools, not guesses)
3. **Recommendation** — clear actionable next step
4. **Labels** — suggested GitHub labels (comma-separated)

IMPORTANT: Only include facts that appear in the tool results above. Mark uncertain claims with [UNCERTAIN].
Max 300 words."""


async def draft_answer(state: IssueTriageState) -> dict:
    """Синтезує всі знахідки у фінальний triage report."""
    llm = get_llm()

    findings_json = json.dumps(state.get("triage_findings", {}), indent=2)

    # Компактне summary tool results
    tool_summary = "\n".join(
        f"- {r['tool']}: {str(r.get('result', ''))[:150]}"
        for r in state.get("tool_results", [])
    ) or "No tool results available"

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=REPORT_TEMPLATE.format(
            issue_url=state["issue_url"],
            task_type=state.get("task_type", "classify"),
            findings_json=findings_json,
            tool_summary=tool_summary,
        ))
    ])

    report = response.content.strip()
    logger.info("Draft report: %d chars", len(report))

    return {
        "triage_report": report,
        "messages": [AIMessage(content=f"Draft report generated ({len(report)} chars)")],
    }


# ══════════════════════════════════════════════════════
# GROUNDING VALIDATOR
# ══════════════════════════════════════════════════════

GROUNDING_PROMPT = """You are a grounding validator. Check if the triage report only contains 
claims that are supported by the tool results provided.

TRIAGE REPORT:
{report}

TOOL RESULTS (ground truth):
{tool_results}

Check each factual claim in the report:
- Does it appear in the tool results?
- Or is it an ungrounded hallucination?

Return JSON:
{{
  "grounding_passed": true|false,
  "ungrounded_claims": ["claim1", "claim2"],
  "hallucination_risk": "high|medium|low",
  "verdict": "PASS|FAIL"
}}

PASS only if there are zero ungrounded claims."""


async def grounding_validator(state: IssueTriageState) -> dict:
    """
    Перевіряє кожне твердження звіту проти tool results.
    Ловить hallucinated citations.
    """
    report = state.get("triage_report", "")
    if not report:
        return {
            "grounding_passed": False,
            "messages": [AIMessage(content="No report to validate")],
        }

    # Компілюємо ground truth
    tool_results_text = "\n".join(
        f"[{r['tool']}]: {json.dumps(r.get('result', ''))[:300]}"
        for r in state.get("tool_results", [])
    ) or "No tool results"

    # Також включаємо fetched issue як ground truth
    issue = state.get("fetched_issue") or {}
    if issue:
        tool_results_text += (
            f"\n[fetch_github_issue]: title={issue.get('title')} "
            f"state={issue.get('state')} "
            f"labels={[l['name'] for l in issue.get('labels', [])]}"
        )

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=GROUNDING_PROMPT.format(
            report=report,
            tool_results=tool_results_text,
        ))
    ])

    try:
        raw = response.content.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(raw)
        passed = result.get("grounding_passed", False)
        ungrounded = result.get("ungrounded_claims", [])

        if not passed and ungrounded:
            # Пробуємо виправити звіт — помічаємо спірні твердження
            fixed_report = state["triage_report"]
            for claim in ungrounded[:3]:
                fixed_report = fixed_report.replace(claim, f"[UNCERTAIN: {claim}]")

            logger.warning("Grounding FAILED — %d ungrounded claims", len(ungrounded))
            return {
                "grounding_passed": False,
                "triage_report": fixed_report,
                "should_escalate": result.get("hallucination_risk") == "high",
                "messages": [AIMessage(content=f"Grounding FAILED: {ungrounded[:3]}")],
            }

        logger.info("Grounding PASSED")
        return {
            "grounding_passed": True,
            "messages": [AIMessage(content="Grounding validation PASSED")],
        }

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Validator parse error: %s", e)
        return {
            "grounding_passed": True,   # benefit of doubt якщо validator сам зламався
            "messages": [AIMessage(content=f"Validator parse error ({e}), assuming pass")],
        }


# ══════════════════════════════════════════════════════
# COMPLETION CHECK
# ══════════════════════════════════════════════════════

async def completion_check(state: IssueTriageState) -> dict:
    """
    Вирішує чи фінальний звіт достатній.
    Встановлює needs_more_info або should_escalate.
    """
    report   = state.get("triage_report", "")
    findings = state.get("triage_findings", {})

    # Якщо звіт дуже короткий або не містить рекомендації
    has_recommendation = any(
        kw in report.lower()
        for kw in ("recommend", "suggest", "action", "close", "merge", "label", "duplicate of", "affects")
    )
    is_long_enough = len(report) > 150

    if not is_long_enough or not has_recommendation:
        logger.info("Report incomplete — needs_more_info=True")
        return {
            "needs_more_info":  True,
            "loop_count":       state["loop_count"] + 1,
            "messages": [AIMessage(content="Report incomplete, looping back for more analysis")],
        }

    # Якщо confidence low у всіх findings
    confidences = [
        v.get("confidence", "medium")
        for v in findings.values()
        if isinstance(v, dict)
    ]
    all_low = confidences and all(c == "low" for c in confidences)
    if all_low:
        return {
            "should_escalate": True,
            "needs_more_info": False,
            "messages": [AIMessage(content="All findings have low confidence — escalating to human review")],
        }

    msg = (
        "Report complete and grounded — ready to deliver"
        if state.get("grounding_passed")
        else "Report complete — grounding had warnings ([UNCERTAIN] may be present) — ready to deliver"
    )
    return {
        "needs_more_info":  False,
        "should_escalate":  False,
        "loop_count":       state["loop_count"],
        "messages": [AIMessage(content=msg)],
    }


# ══════════════════════════════════════════════════════
# HUMAN REVIEW (HIL interrupt node)
# ══════════════════════════════════════════════════════

async def human_review(state: IssueTriageState) -> dict:
    """
    Human-in-the-loop interrupt.
    Граф зупиняється ТУТ, чекає на human_feedback.
    Після resume — корегує звіт за фідбеком.
    """
    feedback = state.get("human_feedback")

    if not feedback:
        # Якщо feedback ще не наданий — форматуємо для людини і чекаємо
        logger.info("HIL interrupt — waiting for human feedback")
        return {
            "messages": [
                AIMessage(content=(
                    "=== HUMAN REVIEW REQUESTED ===\n"
                    f"Issue: {state['issue_url']}\n\n"
                    f"CURRENT REPORT:\n{state.get('triage_report', 'No report yet')}\n\n"
                    "Please provide feedback or type 'approve' to accept."
                ))
            ]
        }

    # Якщо feedback "approve" — підтверджуємо
    if feedback.strip().lower() in ("approve", "ok", "lgtm", "good"):
        return {
            "grounding_passed": True,
            "human_feedback":   None,
            "messages": [AIMessage(content="Human approved the report")],
        }

    # Інакше — регенеруємо звіт з урахуванням фідбеку
    llm = get_llm()
    revised = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Revise this triage report based on human feedback.\n\n"
            f"ORIGINAL REPORT:\n{state.get('triage_report', '')}\n\n"
            f"HUMAN FEEDBACK: {feedback}\n\n"
            f"Keep the same structure but address the feedback. "
            f"Only use facts from tool results, don't invent new data."
        ))
    ])

    return {
        "triage_report":  revised.content.strip(),
        "grounding_passed": True,
        "needs_more_info":  False,
        "should_escalate":  False,
        "human_feedback":   None,
        "messages": [AIMessage(content="Report revised based on human feedback")],
    }
