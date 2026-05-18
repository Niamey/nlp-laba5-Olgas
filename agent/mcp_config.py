"""
Спільна конфігурація MCP для main.py та утиліт (inventory, тести).

Підключені MCP-сервери (два процеси stdio → `MultiServerMCPClient` у `main.py`):

┌─────────────────┬──────────────────────────────────────────────────────────────────┐
│ Ключ у конфігу  │ Що це                                                             │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ github_triage   │ НАШ сервер: `mcp_server/server.py` (FastMCP). Тулі GitHub issue,  │
│                 │ пошук схожих, GitHub API URL, локальні triage-нотатки.            │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ fetch           │ СТОРОННІЙ сервер: пакет `mcp-server-fetch` (див. `.env.example`). │
│                 │ Увімкнено лише якщо задано `FETCH_MCP_COMMAND` (+ `FETCH_MCP_ARGS`).│
│                 │ Один тул `fetch` — довільний HTTP / markdown.                      │
└─────────────────┴──────────────────────────────────────────────────────────────────┘

Повертає `dict[str, dict]` — саме цей словник передається в `MultiServerMCPClient(...)`.
Див. також константу `MCP_SERVERS_OVERVIEW` нижче — зручно показати на захисті в IDE.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from agent.paths import cache_directory_env_value

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

# Явний перелік для доповіді: (ключ у MultiServerMCPClient, шлях або пакет, короткий опис)
MCP_SERVERS_OVERVIEW: tuple[tuple[str, str, str], ...] = (
    (
        "github_triage",
        str(_ROOT / "mcp_server" / "server.py"),
        "власний FastMCP: fetch_github_issue, fetch_url, search_similar_issues, save/get triage notes",
    ),
    (
        "fetch",
        "mcp-server-fetch (команда з FETCH_MCP_COMMAND, напр. `python -m mcp_server_fetch`)",
        "сторонній MCP: один інструмент `fetch` (HTTP → markdown)",
    ),
)


def repo_root() -> Path:
    return _ROOT


def build_mcp_connections(*, warn_if_no_fetch: bool = True) -> dict[str, dict]:
    """
    Збірка підключень для langchain_mcp_adapters.MultiServerMCPClient.

    Сервер 1 (завжди): github_triage — див. `MCP_SERVERS_OVERVIEW[0]`.
    Сервер 2 (опційно): fetch — див. `MCP_SERVERS_OVERVIEW[1]` + змінні середовища FETCH_*.
    """
    # ─── MCP server 1: github_triage (project-local FastMCP) ─────────────────
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

    # ─── MCP server 2: fetch (third-party, optional) ─────────────────────────
    fetch_server = os.getenv("FETCH_MCP_COMMAND", "").strip()
    fetch_extra = os.getenv("FETCH_MCP_ARGS", "").strip().split()
    if fetch_server:
        # Без жорсткого C:\Users\... — той самий інтерпретатор, що запускає main.py (зазвичай .venv).
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
