"""Shared paths (CACHE_DIR fallback for main process and MCP subprocess)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def cache_directory() -> Path:
    """
    MCP SQLite cache directory. Honors CACHE_DIR; otherwise OS temp dir + triage_cache.
    Empty CACHE_DIR counts as unset.
    """
    raw = os.environ.get("CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "triage_cache"


def cache_directory_env_value() -> str:
    """String form for MCP child process env."""
    return str(cache_directory())
