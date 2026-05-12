#!/usr/bin/env python3
"""
Список усіх налаштованих MCP-серверів і тулі з описами.

    python scripts/mcp_inventory.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from agent.mcp_inventory import print_mcp_inventory


if __name__ == "__main__":
    asyncio.run(print_mcp_inventory())
