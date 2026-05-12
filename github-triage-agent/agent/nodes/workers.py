"""
workers.py — 4 спеціалізованих вузли тріажу.
Кожен використовує відповідні MCP інструменти і LLM.
"""
from __future__ import annotations
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import IssueTriageState
from agent.llm import get_llm, SYSTEM_PROMPT
from agent.mcp_tools_context import get_mcp_tools
from agent.mcp_unwrap import unwrap_mcp_json_payload

logger = logging.getLogger(__name__)


def _get_tools(config: RunnableConfig) -> dict:
    t = (config.get("configurable") or {}).get("tools") or {}
    return t if t else get_mcp_tools()


def _json_from_str(s: str) -> dict | list | None:
    s = (s or "").strip()
    if not s.startswith(("{", "[")):
        return None
    try:
        out = json.loads(s)
        return out if isinstance(out, (dict, list)) else None
    except json.JSONDecodeError:
        return None


def _parse_fetch_url_response(
    raw,
    *,
    expect: str = "dict",
    _depth: int = 0,
) -> dict | list | None:
    """
    Розбір тіла відповіді fetch_url (MCP: рядок JSON, {type,text}, list блоків).

    expect="dict" — відповіді на кшталт git/trees (об'єкт з ключем "tree").
    expect="list" — масив з GitHub (наприклад comments), без рекурсії по кожному елементу.
    """
    if _depth > 8:
        return None

    out = unwrap_mcp_json_payload(raw)

    if isinstance(out, str):
        return _json_from_str(out)

    if isinstance(out, dict) and out.get("type") == "text" and isinstance(out.get("text"), str):
        return _parse_fetch_url_response(out["text"], expect=expect, _depth=_depth + 1)

    # Вже масив об'єктів GitHub (коментарі тощо), а не список MCP-блоків
    if (
        isinstance(out, list)
        and out
        and expect == "list"
        and all(isinstance(x, dict) for x in out)
        and not (len(out) == 1 and out[0].get("type") == "text")
    ):
        return out

    if isinstance(out, list):
        for item in out:
            got = _parse_fetch_url_response(item, expect=expect, _depth=_depth + 1)
            if got is not None:
                return got
        return None

    if isinstance(out, dict):
        if expect == "dict":
            return out
        if expect == "list" and "tree" not in out:
            # Один об'єкт замість масиву — перетворюємо на список
            return [out]
        return out

    return None


def _issue_summary(state: IssueTriageState) -> str:
    """Короткий опис issue для промптів."""
    issue = state.get("fetched_issue") or {}
    return (
        f"Issue #{issue.get('number')}: {issue.get('title', 'N/A')}\n"
        f"Body: {(issue.get('body') or '')[:1500]}"
    )


# ──────────────────────────────────────────────────────
# WORKER 1: Duplicate Detection
# ──────────────────────────────────────────────────────

async def duplicate_point(state: IssueTriageState, config: RunnableConfig) -> dict:
    """
    Шукає дублікати issue в репозиторії.
    Стратегія: 3 різних пошукових запити (title keywords, error message, label-based).
    """
    tools = _get_tools(config)
    search_tool = tools.get("search_similar_issues")
    issue = state.get("fetched_issue") or {}
    repo = f"{state['repo_owner']}/{state['repo_name']}"
    title = issue.get("title", "")
    body  = (issue.get("body") or "")[:500]

    new_tool_results = []
    similar_issues = []

    if search_tool:
        # Запит 1: за ключовими словами заголовку
        queries = [
            title[:80],
            # Запит 2: за помилкою з тіла (якщо є)
            _extract_error(body),
            # Запит 3: за лейблами
            " ".join(l["name"] for l in issue.get("labels", [])[:3]),
        ]
        queries = [q for q in queries if q.strip()]

        for query in queries[:3]:
            try:
                raw = await search_tool.ainvoke({
                    "repo": repo,
                    "query": query,
                    "max_results": 5,
                })
                parsed = unwrap_mcp_json_payload(raw)
                if isinstance(parsed, str):
                    try:
                        results = json.loads(parsed)
                    except json.JSONDecodeError:
                        results = []
                elif isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = [] if parsed.get("error") else [parsed]
                else:
                    results = []
                if not isinstance(results, list):
                    results = []
                # Фільтруємо сам issue; ігноруємо не-dict елементи
                filtered = [
                    r for r in results
                    if isinstance(r, dict)
                    and r.get("number") is not None
                    and r.get("number") != issue.get("number")
                ]
                similar_issues.extend(filtered[:3])
                new_tool_results.append({
                    "tool": "search_similar_issues",
                    "args": {"repo": repo, "query": query},
                    "result": results,
                })
            except Exception as e:
                logger.warning("Search failed for query '%s': %s", query[:40], e)

    # LLM аналізує знайдені issues
    llm = get_llm()
    similar_text = json.dumps(
        [{"number": s.get("number"), "title": s.get("title"), "state": s.get("state")}
         for s in similar_issues[:5]],
        indent=2
    )

    analysis = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Analyze if this issue is a duplicate:\n\n"
            f"{_issue_summary(state)}\n\n"
            f"Potentially similar issues found:\n{similar_text}\n\n"
            f"For each similar issue, determine if it's a true duplicate, related, or different.\n"
            f"Return JSON: {{\"is_duplicate\": bool, \"duplicate_of\": number_or_null, "
            f"\"confidence\": \"high|medium|low\", \"related_issues\": [list of numbers], "
            f"\"reasoning\": \"explanation\"}}"
        ))
    ])

    try:
        raw = analysis.content.strip().lstrip("```json").rstrip("```").strip()
        findings = json.loads(raw)
    except Exception:
        findings = {"is_duplicate": False, "reasoning": analysis.content, "confidence": "low"}

    return {
        "similar_issues": similar_issues,
        "tool_results": state["tool_results"] + new_tool_results,
        "tool_calls_count": state["tool_calls_count"] + len(new_tool_results),
        "triage_findings": {**state.get("triage_findings", {}), "duplicate_analysis": findings},
        "messages": [AIMessage(content=f"Duplicate analysis: {json.dumps(findings, indent=2)}")],
    }


# ──────────────────────────────────────────────────────
# WORKER 2: Code Area Identification
# ──────────────────────────────────────────────────────

async def code_area_point(state: IssueTriageState, config: RunnableConfig) -> dict:
    """
    Визначає яка частина кодової бази відповідає за issue.
    Без запуску коду — лише текстовий аналіз.
    """
    tools = _get_tools(config)
    llm   = get_llm()
    issue = state.get("fetched_issue") or {}

    # Намагаємось отримати структуру репо (якщо fetch tool є)
    fetch_tool = tools.get("fetch_url")
    repo_info  = ""
    new_results = []

    if fetch_tool:
        api_url = (
            f"https://api.github.com/repos/{state['repo_owner']}/{state['repo_name']}/git/trees/HEAD"
            "?recursive=1"
        )
        try:
            raw = await fetch_tool.ainvoke({"url": api_url})
            data = _parse_fetch_url_response(raw, expect="dict")
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object from trees API, got {type(data).__name__}")
            files = [
                t["path"] for t in data.get("tree", [])
                if isinstance(t, dict) and t.get("type") == "blob" and str(t.get("path", "")).endswith(".py")
            ]
            # Показуємо тільки топ-рівень + файли з ключовими словами
            keywords = _extract_keywords(issue.get("title", "") + " " + (issue.get("body") or ""))
            relevant = [f for f in files if any(kw in f.lower() for kw in keywords)]
            repo_info = "\n".join(relevant[:30] or files[:30])
            new_results.append({"tool": "fetch_url", "args": {"url": api_url}, "result": f"{len(files)} files"})
        except Exception as e:
            logger.info("Could not fetch repo tree: %s", e)
            repo_info = "Repository structure not available"

    # LLM визначає code area
    analysis = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Identify the code area affected by this issue.\n\n"
            f"{_issue_summary(state)}\n\n"
            f"Repository Python files:\n{repo_info or 'not available'}\n\n"
            f"Based on the issue description, identify:\n"
            f"- Which module/component is most likely affected\n"
            f"- Specific file paths if identifiable\n"
            f"- Confidence level\n\n"
            f"Return JSON: {{\"affected_modules\": [\"path1\", \"path2\"], "
            f"\"component\": \"name\", \"confidence\": \"high|medium|low\", "
            f"\"reasoning\": \"explanation\"}}"
        ))
    ])

    try:
        raw = analysis.content.strip().lstrip("```json").rstrip("```").strip()
        findings = json.loads(raw)
    except Exception:
        findings = {"component": "unknown", "reasoning": analysis.content, "confidence": "low"}

    return {
        "repo_structure": repo_info,
        "tool_results": state["tool_results"] + new_results,
        "tool_calls_count": state["tool_calls_count"] + len(new_results) + 1,
        "triage_findings": {**state.get("triage_findings", {}), "code_area": findings},
        "messages": [AIMessage(content=f"Code area: {json.dumps(findings, indent=2)}")],
    }


# ──────────────────────────────────────────────────────
# WORKER 3: Stale Issue Analysis
# ──────────────────────────────────────────────────────

async def stale_point(state: IssueTriageState, config: RunnableConfig) -> dict:
    """
    Аналізує stale issue: активність, заблоковані питання, наступний крок.
    """
    tools  = _get_tools(config)
    llm    = get_llm()
    issue  = state.get("fetched_issue") or {}

    # Отримуємо коментарі якщо є fetch tool
    comments_data = []
    new_results   = []
    fetch_tool    = tools.get("fetch_url")

    comments_url = issue.get("comments_url", "")
    if fetch_tool and comments_url and issue.get("comments", 0) > 0:
        try:
            raw = await fetch_tool.ainvoke({"url": comments_url})
            parsed = _parse_fetch_url_response(raw, expect="list")
            comments_data = parsed if isinstance(parsed, list) else []
            new_results.append({
                "tool": "fetch_url",
                "args": {"url": comments_url},
                "result": f"{len(comments_data)} comments",
            })
        except Exception as e:
            logger.info("Could not fetch comments: %s", e)

    # Форматуємо останні 5 коментарів
    recent_comments = ""
    if comments_data:
        for c in comments_data[-5:]:
            if not isinstance(c, dict):
                continue
            author = c.get("user", {}).get("login", "?")
            date   = c.get("updated_at", "?")[:10]
            body   = (c.get("body") or "")[:300]
            recent_comments += f"\n[{date}] @{author}: {body}\n"

    analysis = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Analyze this stale GitHub issue:\n\n"
            f"{_issue_summary(state)}\n"
            f"Created: {issue.get('created_at', 'N/A')}\n"
            f"Last updated: {issue.get('updated_at', 'N/A')}\n\n"
            f"Recent comments:{recent_comments or ' none'}\n\n"
            f"Analyze:\n"
            f"1. Current status (still relevant? blocked? waiting for author?)\n"
            f"2. Outstanding questions\n"
            f"3. Recommended next action\n\n"
            f"Return JSON: {{\"still_relevant\": bool, \"blocking_reason\": \"string\", "
            f"\"outstanding_questions\": [\"q1\", \"q2\"], "
            f"\"recommended_action\": \"close|ping_author|needs_pr|needs_info|keep_open\", "
            f"\"reasoning\": \"explanation\"}}"
        ))
    ])

    try:
        raw = analysis.content.strip().lstrip("```json").rstrip("```").strip()
        findings = json.loads(raw)
    except Exception:
        findings = {"still_relevant": True, "reasoning": analysis.content}

    return {
        "tool_results": state["tool_results"] + new_results,
        "tool_calls_count": state["tool_calls_count"] + len(new_results) + 1,
        "triage_findings": {**state.get("triage_findings", {}), "stale_analysis": findings},
        "messages": [AIMessage(content=f"Stale analysis: {json.dumps(findings, indent=2)}")],
    }


# ──────────────────────────────────────────────────────
# WORKER 4: General Classification
# ──────────────────────────────────────────────────────

async def classify_point(state: IssueTriageState, config: RunnableConfig) -> dict:
    """
    Загальна класифікація issue: bug / feature / question / docs / duplicate.
    Також виставляє пріоритет і тег.
    """
    tools     = _get_tools(config)
    save_tool = tools.get("save_triage_note")
    llm       = get_llm()
    issue     = state.get("fetched_issue") or {}
    new_results = []

    analysis = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Classify this GitHub issue:\n\n"
            f"{_issue_summary(state)}\n\n"
            f"Current labels: {[l['name'] for l in issue.get('labels', [])]}\n\n"
            f"Classify into:\n"
            f"- Type: bug | feature | question | documentation | duplicate\n"
            f"- Priority: critical | high | medium | low\n"
            f"- Confidence: high | medium | low\n\n"
            f"Return JSON: {{\"type\": \"...\", \"priority\": \"...\", "
            f"\"confidence\": \"...\", \"suggested_labels\": [\"l1\", \"l2\"], "
            f"\"justification\": \"cite specific content from the issue\"}}"
        ))
    ])

    try:
        raw = analysis.content.strip().lstrip("```json").rstrip("```").strip()
        findings = json.loads(raw)
    except Exception:
        findings = {"type": "question", "reasoning": analysis.content, "confidence": "low"}

    # Зберігаємо нотатку в MCP (local bookkeeping)
    if save_tool:
        try:
            await save_tool.ainvoke({
                "issue_url": state["issue_url"],
                "note": f"Classification: {findings.get('type')} ({findings.get('confidence')})",
            })
            new_results.append({
                "tool": "save_triage_note",
                "args": {"issue_url": state["issue_url"]},
                "result": "saved",
            })
        except Exception as e:
            logger.warning("Could not save note: %s", e)

    return {
        "tool_results": state["tool_results"] + new_results,
        "tool_calls_count": state["tool_calls_count"] + len(new_results),
        "triage_findings": {**state.get("triage_findings", {}), "classification": findings},
        "messages": [AIMessage(content=f"Classification: {json.dumps(findings, indent=2)}")],
    }


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _extract_error(text: str) -> str:
    """Витягує першу помилку з тексту (для пошуку дублікатів)."""
    import re
    patterns = [
        r"(Error|Exception|Traceback)[^\n]{0,100}",
        r"`([^`]{10,80})`",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)[:80]
    return text[:60]


def _extract_keywords(text: str) -> list[str]:
    """Витягує ключові слова з тексту для пошуку файлів."""
    import re
    words = re.findall(r"\b[a-z][a-z_]{3,}\b", text.lower())
    stopwords = {"this", "that", "with", "from", "have", "when", "what", "will", "been", "also"}
    return list(set(w for w in words if w not in stopwords))[:10]
