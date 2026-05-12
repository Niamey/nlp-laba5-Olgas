"""
fetch_issue_node — отримує дані issue через MCP інструмент fetch_github_issue.
fallback: GitHub REST у цьому процесі (LangGraph+MCP часто не доставляють тулі в вузол).
"""
from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.state import IssueTriageState
from agent.mcp_tools_context import get_mcp_tools
from agent.github_direct import fetch_github_issue_direct
from agent.mcp_unwrap import unwrap_mcp_json_payload

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _tools_for_run(config: RunnableConfig) -> dict:
    t = (config.get("configurable") or {}).get("tools") or {}
    return t if t else get_mcp_tools()


def _normalize_issue_payload(raw: Any) -> dict | None:
    """MCP часом обгортає issue в list з одним dict — розгортаємо."""
    if raw is None:
        return None
    raw = unwrap_mcp_json_payload(raw)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        return raw[0]
    if isinstance(raw, list):
        logger.warning("MCP fetch_github_issue повернув list (len=%d), очікуємо один dict", len(raw))
        return None
    logger.warning("MCP fetch_github_issue повернув %s, очікуємо dict", type(raw).__name__)
    return None


def _issue_payload_usable(d: dict) -> bool:
    """True якщо це реальна відповідь GitHub або коректний 404, а не сміття/обгортка MCP."""
    if d.get("error"):
        return False
    if d.get("message") == "Not Found":
        return True
    return d.get("number") is not None


def _finalize_fetch(state: IssueTriageState, url: str, issue_data: dict) -> dict:
    """Один успішний payload issue → стан."""
    if issue_data.get("message") == "Not Found":
        return {
            "last_error": f"Issue not found: {url}",
            "error_count": state["error_count"] + 1,
            "task_type": "not_found",
            "messages": [AIMessage(content=f"Issue {url} does not exist (404)")],
        }
    if issue_data.get("error"):
        return {
            "last_error": str(issue_data.get("error")),
            "error_count": state["error_count"] + 1,
        }

    tool_result = {
        "tool": "fetch_github_issue",
        "args": {"url": url},
        "result": issue_data,
    }
    logger.info(
        "Fetched issue #%s: '%s' [%s] via=%s",
        issue_data.get("number"),
        (issue_data.get("title") or "?")[:60],
        issue_data.get("state", "?"),
        issue_data.get("_via", "mcp"),
    )
    return {
        "fetched_issue": issue_data,
        "tool_results": state["tool_results"] + [tool_result],
        "tool_calls_count": state["tool_calls_count"] + 1,
        "messages": [
            AIMessage(content=(
                f"Fetched issue #{issue_data.get('number')}: "
                f"'{issue_data.get('title', 'no title')}'\n"
                f"State: {issue_data.get('state')} | "
                f"Labels: {[l['name'] for l in issue_data.get('labels', [])]} | "
                f"Comments: {issue_data.get('comments', 0)}"
            ))
        ],
    }


async def fetch_issue_node(state: IssueTriageState, config: RunnableConfig) -> dict:
    url = state["issue_url"]
    tools: dict = _tools_for_run(config)
    fetch_tool = tools.get("fetch_github_issue")
    logger.info(
        "fetch_issue_node: %d tools, fetch_github_issue=%s",
        len(tools),
        "ok" if fetch_tool else "MISSING",
    )

    issue_data: dict | None = None

    if fetch_tool:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                result = await fetch_tool.ainvoke({"url": url})
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        parsed = result
                else:
                    parsed = result
                issue_data = _normalize_issue_payload(parsed)
                break
            except json.JSONDecodeError as e:
                logger.warning("MCP fetch JSON error: %s — trying direct HTTP", e)
                break
            except Exception as e:
                err_str = str(e)
                retries += 1
                if any(kw in err_str.lower() for kw in ("rate limit", "timeout", "connect", "502", "503")):
                    logger.warning("MCP fetch retriable (%d/%d): %s", retries, MAX_RETRIES, err_str)
                    if retries < MAX_RETRIES:
                        import asyncio
                        await asyncio.sleep(2 ** retries)
                        continue
                logger.warning("MCP fetch failed: %s — trying direct HTTP", err_str)
                break

    need_direct = issue_data is None or not _issue_payload_usable(issue_data)
    if need_direct:
        if issue_data is not None:
            logger.info("MCP payload непридатний (error або без number) — direct HTTP для %s", url)
        else:
            logger.info("Using direct GitHub REST for %s", url)
        issue_data = await fetch_github_issue_direct(url)

    if not isinstance(issue_data, dict):
        logger.error("fetch_issue_node: очікуваний dict issue після MCP/HTTP, отримано %s", type(issue_data).__name__)
        return {
            "last_error": "Unexpected issue payload shape from GitHub",
            "error_count": state["error_count"] + 1,
        }

    return _finalize_fetch(state, url, issue_data)
