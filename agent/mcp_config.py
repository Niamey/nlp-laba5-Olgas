"""
Спільна конфігурація MCP для main.py та утиліт (inventory, тести).

Повертає dict з іменами серверів як ключами (github_triage, опціонально fetch).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from agent.paths import cache_directory_env_value

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _ROOT


def build_mcp_connections(*, warn_if_no_fetch: bool = True) -> dict[str, dict]:
    """
    Збірка підключень для langchain_mcp_adapters.MultiServerMCPClient.

    Сервери:
    - github_triage — власний FastMCP (mcp_server/server.py)
    - fetch — опціонально, mcp-server-fetch (FETCH_MCP_COMMAND / FETCH_MCP_ARGS)
    """
    cfg: dict[str, dict] = {
        "github_triage": {
            "command": sys.executable,
            "args": [str(_ROOT / "mcp_server" / "server.py")],
            "transport": "stdio",
            "env": {
                "GITHUB_TOKEN": (os.getenv("GITHUB_TOKEN") or "").strip(),
                "CACHE_DIR": cache_directory_env_value(),
            },
        },
    }

    fetch_server = os.getenv("FETCH_MCP_COMMAND", "").strip()
    fetch_extra = os.getenv("FETCH_MCP_ARGS", "").strip().split()
    if fetch_server:
        if fetch_server.upper() == "AUTO":
            fetch_server = sys.executable
            logger.info("FETCH_MCP_COMMAND=AUTO → using sys.executable: %s", fetch_server)
        if not fetch_extra and fetch_server.lower() in {"uvx", "uv"}:
            fetch_extra = ["mcp-server-fetch"]
            logger.info(
                "FETCH_MCP_ARGS not set — defaulting to package mcp-server-fetch for %s",
                fetch_server,
            )
        cfg["fetch"] = {
            "command": fetch_server,
            "args": fetch_extra,
            "transport": "stdio",
        }
    elif warn_if_no_fetch:
        logger.warning(
            "Fetch MCP server disabled. Set FETCH_MCP_COMMAND and FETCH_MCP_ARGS "
            "(e.g. uvx + mcp-server-fetch). Using github_triage fetch_url only."
        )

    return cfg


def mcp_config_for_main() -> dict[str, dict]:
    """Alias для main.py."""
    return build_mcp_connections(warn_if_no_fetch=True)
