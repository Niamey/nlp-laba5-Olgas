"""
drafter.py    — синтезує знахідки у структурований звіт
validator.py  — перевіряє grounding (кожне твердження = tool result)
completion.py — вирішує done/more/escalate + human_review interrupt
"""
from __future__ import annotations
import json
import logging
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import IssueTriageState
from agent.llm import get_llm, SYSTEM_PROMPT
import os

# Достатньо тексту для драфтеру й валідатора (раніше 150/300 символів різало body issue → хибні FAIL)
_MAX_TOOL_SUMMARY_CHARS = int(os.environ.get("TRIAGE_TOOL_SUMMARY_CHARS", "2500"))
_MAX_VALIDATION_CHARS_PER_TOOL = int(os.environ.get("TRIAGE_GROUND_TOOL_CHARS", "12000"))
_MAX_ISSUE_BODY_IN_GROUNDING = int(os.environ.get("TRIAGE_GROUND_ISSUE_BODY", "14000"))
_MAX_TOTAL_GROUNDING_BLOB = int(os.environ.get("TRIAGE_GROUND_TOTAL", "52000"))
_MAX_FINDINGS_IN_GROUNDING = int(os.environ.get("TRIAGE_GROUND_FINDINGS", "16000"))

logger = logging.getLogger(__name__)


def _clip_for_prompt(obj: object, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    raw = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n…[truncated]…"


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str:
    """Виділяє перший сбалансований {...} з відповіді LLM (текст до/після JSON ігнорується)."""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()


def _coerce_bool(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "pass")
    return default


def _parse_validator_json(content: str) -> dict:
    """Парсить JSON від grounding-LLM; кілька стратегій, бо відповідь часто з markdown/прологом."""
    raw = _strip_code_fences(content)
    last_err: Exception | None = None
    for candidate in (raw, _extract_first_json_object(raw)):
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(out, dict):
            return out
        last_err = TypeError("validator JSON root is not an object")
    if isinstance(last_err, json.JSONDecodeError):
        raise last_err
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("Could not parse validator JSON", raw or "", 0)


def _build_ground_truth_blob(state: IssueTriageState) -> str:
    """Повний контекст для валідації: triage_findings + tool results + body issue."""
    parts: list[str] = []

    findings = state.get("triage_findings")
    if findings:
        parts.append(
            "[triage_findings] (structured output from workers — same pipeline as the report):\n"
            + _clip_for_prompt(findings, _MAX_FINDINGS_IN_GROUNDING)
        )

    for r in state.get("tool_results", []):
        name = r.get("tool", "?")
        payload = r.get("result", "")
        chunk = _clip_for_prompt(payload, _MAX_VALIDATION_CHARS_PER_TOOL)
        parts.append(f"[{name}]:\n{chunk}")

    issue = state.get("fetched_issue") or {}
    if issue and not issue.get("error") and issue.get("message") != "Not Found":
        slim = {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "state": issue.get("state"),
            "body": (issue.get("body") or "")[:_MAX_ISSUE_BODY_IN_GROUNDING],
            "labels": [l.get("name") for l in issue.get("labels") or [] if isinstance(l, dict)],
            "comments": issue.get("comments"),
            "user": (issue.get("user") or {}).get("login") if isinstance(issue.get("user"), dict) else None,
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
        }
        parts.append("[fetched_issue_canonical]:\n" + json.dumps(slim, ensure_ascii=False, default=str))

    blob = "\n\n---\n\n".join(parts) if parts else "No tool results"
    if len(blob) > _MAX_TOTAL_GROUNDING_BLOB:
        return blob[: _MAX_TOTAL_GROUNDING_BLOB - 30] + "\n…[grounding blob truncated]…"
    return blob


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
Use plain **markdown** only. Do NOT wrap the entire report in a ```json``` or ``` code fence.
Max 300 words."""


async def draft_answer(state: IssueTriageState) -> dict:
    """Синтезує всі знахідки у фінальний triage report."""
    llm = get_llm()

    findings_json = json.dumps(state.get("triage_findings", {}), indent=2)

    # Компактне summary tool results (повинно містити ключові цитати/факти, не обрізати issue до заголовка)
    tool_summary = "\n".join(
        f"- {r['tool']}: {_clip_for_prompt(r.get('result', ''), _MAX_TOOL_SUMMARY_CHARS)}"
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

    report = _strip_code_fences(response.content.strip())
    logger.info("Draft report: %d chars", len(report))

    return {
        "triage_report": report,
        "messages": [AIMessage(content=f"Draft report generated ({len(report)} chars)")],
    }


# ══════════════════════════════════════════════════════
# GROUNDING VALIDATOR
# ══════════════════════════════════════════════════════

GROUNDING_PROMPT = """You are a grounding validator. The TOOL RESULTS block is the only ground truth.

TRIAGE REPORT:
{report}

TOOL RESULTS (ground truth — includes triage_findings, full tool payloads, and issue title/body when fetched):
{tool_results}

Rules:
- The [triage_findings] section is part of ground truth: if the report repeats issue type, confidence, priority, or reasoning that appears there OR in the issue body OR in tool payloads, it is GROUNDED (even if phrased differently).
- Paraphrases and summaries of facts that ARE present in the tool results count as GROUNDED.
- Obvious logical consequences of stated facts (e.g. "likely a support question" when maintainers said so) are GROUNDED if tied to quoted/stated evidence.
- Mark as UNGROUNDED only if the report states specific facts, numbers, usernames, file paths, error messages, or timelines that do NOT appear in the blocks above (hallucination / invention).
- Sections like **Labels** are suggestions: label names need not appear verbatim in tools if they are reasonable interpretations of the described issue.

Return ONLY one JSON object (strict JSON: lowercase true/false/null, double-quoted keys). No markdown fences, no text before or after the JSON.

{{
  "grounding_passed": true|false,
  "ungrounded_claims": ["claim1", "claim2"],
  "hallucination_risk": "high|medium|low",
  "verdict": "PASS|FAIL"
}}

PASS (grounding_passed true) only if there are zero ungrounded claims by the rules above."""


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

    tool_results_text = _build_ground_truth_blob(state)

    llm = get_llm(temperature=0.0)
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=GROUNDING_PROMPT.format(
            report=report,
            tool_results=tool_results_text,
        ))
    ])

    try:
        result = _parse_validator_json(response.content)
        passed = _coerce_bool(result.get("grounding_passed"), False)
        verdict = str(result.get("verdict") or "").strip().upper()
        if verdict == "FAIL":
            passed = False
        ungrounded = result.get("ungrounded_claims") or []
        if not isinstance(ungrounded, list):
            ungrounded = []

        if not passed:
            fixed_report = state["triage_report"]
            for claim in ungrounded[:3]:
                if isinstance(claim, str) and claim.strip():
                    fixed_report = fixed_report.replace(claim, f"[UNCERTAIN: {claim}]")

            logger.warning(
                "Grounding FAILED — passed=%s, ungrounded=%s",
                passed,
                ungrounded[:5],
            )
            return {
                "grounding_passed": False,
                "triage_report": fixed_report,
                "should_escalate": result.get("hallucination_risk") == "high",
                "messages": [
                    AIMessage(
                        content=(
                            f"Grounding FAILED: {ungrounded[:3]}"
                            if ungrounded
                            else "Grounding FAILED (validator marked fail with no claim list)"
                        )
                    )
                ],
            }

        logger.info("Grounding PASSED")
        return {
            "grounding_passed": True,
            "messages": [AIMessage(content="Grounding validation PASSED")],
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # Чесно: без валідного JSON від валідатора автоматично підтвердити grounding неможливо
        logger.warning("Validator parse error: %s", e)
        return {
            "grounding_passed": False,
            "messages": [
                AIMessage(
                    content=(
                        f"Grounding validator returned non-JSON or invalid output ({e!s}); "
                        "grounding_passed=false (inconclusive — re-run or inspect report manually)"
                    )
                )
            ],
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
