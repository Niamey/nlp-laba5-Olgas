"""Розгортання відповідей MCP від langchain-mcp-adapters (облік {type:text, text:JSON})."""
from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


def unwrap_mcp_json_payload(raw: Any) -> Any:
    """
    Клієнт MCP часто віддає {'type': 'text', 'text': '<json>'} або список з одного такого блоку.
    `text` може бути об'єктом {...} або масивом [...] у вигляді рядка.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return raw
        return raw
    if isinstance(raw, list) and len(raw) == 1:
        return unwrap_mcp_json_payload(raw[0])
    if isinstance(raw, dict) and raw.get("type") == "text":
        blob = raw.get("text")
        if isinstance(blob, str):
            s = blob.strip()
            if s.startswith(("{", "[")):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    _logger.warning("MCP text payload не JSON (prefix=%s)", s[:80])
                    return raw
        return raw
    return raw
