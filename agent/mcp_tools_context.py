"""
MCP-тулси для поточного прогону `graph.ainvoke`.

Не ContextVar: LangGraph викликає вузли через asyncio.create_task(..., context=...),
і окремий context НЕ успадковує наші ContextVar із run_triage — тулі там «губилися».
"""
from __future__ import annotations

from typing import Any

_session: dict[str, Any] | None = None


def set_mcp_tools(tools: dict[str, Any]) -> None:
    global _session
    _session = dict(tools)


def get_mcp_tools() -> dict[str, Any]:
    return dict(_session) if _session is not None else {}
