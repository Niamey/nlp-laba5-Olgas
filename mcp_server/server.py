#!/usr/bin/env python3
"""
GitHub Triage MCP Server — власний MCP сервер (окремий процес).

У конфігурації агента (`agent/mcp_config.py`) цей файл підключений під ключем **github_triage**
(перший з двох MCP; другий — опційний **fetch** з пакета mcp-server-fetch, без цього репо).

Запуск:
    python mcp_server/server.py

Інструменти:
    1. fetch_github_issue   — завантажує issue з GitHub API (з кешем)
    2. search_similar_issues — шукає схожі issues (з кешем)
    3. fetch_url             — завантажує довільний URL (для API calls)
    4. save_triage_note      — зберігає нотатку (local bookkeeping)
    5. get_triage_notes      — читає нотатки (local bookkeeping)
"""
import os
import json
import sqlite3
import asyncio
import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ── Конфігурація ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MCP] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
def _cache_dir() -> Path:
    raw = os.getenv("CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "triage_cache"


CACHE_DIR = _cache_dir()
CACHE_TTL    = int(os.getenv("CACHE_TTL_SECONDS", str(3600 * 6)))   # 6 годин
RATE_LIMIT_DELAY = 0.5   # секунди між запитами до GitHub API

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "triage.db"

# ── SQLite ─────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key      TEXT PRIMARY KEY,
            value    TEXT NOT NULL,
            ts       REAL NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_url  TEXT NOT NULL,
            note       TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    db.commit()
    return db


def _cache_get(key: str) -> Optional[str]:
    """Читає з кешу якщо не прострочений."""
    try:
        db = _get_db()
        row = db.execute(
            "SELECT value, ts FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row and (time.time() - row[1]) < CACHE_TTL:
            logger.info("Cache HIT: %s", key[:60])
            return row[0]
        logger.info("Cache MISS: %s", key[:60])
        return None
    except Exception as e:
        logger.warning("Cache read error: %s", e)
        return None


def _cache_set(key: str, value: str) -> None:
    """Зберігає в кеш."""
    try:
        db = _get_db()
        db.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, value, time.time())
        )
        db.commit()
    except Exception as e:
        logger.warning("Cache write error: %s", e)


def _make_key(prefix: str, data: str) -> str:
    h = hashlib.sha256(data.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"


# ── HTTP client ────────────────────────────────────────────────────
def _github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "github-triage-agent/1.0",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


# ── FastMCP сервер ─────────────────────────────────────────────────
mcp = FastMCP(
    "github-triage",
    instructions=(
        "GitHub issue triage tools. Use fetch_github_issue to get issue data, "
        "search_similar_issues to find duplicates, fetch_url for any GitHub API call, "
        "and save_triage_note / get_triage_notes for local bookkeeping."
    ),
)


# ══════════════════════════════════════════════════════════════════
# Tool 1: fetch_github_issue
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_github_issue(url: str) -> str:
    """
    Fetch a GitHub issue including title, body, labels, state, comments count,
    created/updated dates, and author. Results are cached for 6 hours.

    Args:
        url: Full GitHub issue URL, e.g. https://github.com/owner/repo/issues/123

    Returns:
        JSON string with issue data or error message.
    """
    import re
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)", url.strip())
    if not match:
        return json.dumps({"error": f"Invalid GitHub issue URL: {url}"})

    owner, repo, number = match.group(1), match.group(2), match.group(3)
    cache_key = _make_key("issue", f"{owner}/{repo}/{number}")

    cached = _cache_get(cache_key)
    if cached:
        return cached

    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    await asyncio.sleep(RATE_LIMIT_DELAY)

    try:
        # Репозиторії GitHub часом переїжджають → 301; інакше httpx.raise_for_status() кидає про redirect.
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(api_url, headers=_github_headers())

        if resp.status_code == 404:
            result = json.dumps({"message": "Not Found", "url": url})
        elif resp.status_code == 403:
            result = json.dumps({"error": "Rate limited or unauthorized", "status": 403})
        else:
            resp.raise_for_status()
            data = resp.json()
            # Компактний формат — залишаємо тільки корисні поля
            result = json.dumps({
                "number":      data.get("number"),
                "title":       data.get("title"),
                "body":        data.get("body"),
                "state":       data.get("state"),
                "labels":      [{"name": l["name"]} for l in data.get("labels", [])],
                "user":        {"login": data.get("user", {}).get("login")},
                "created_at":  data.get("created_at"),
                "updated_at":  data.get("updated_at"),
                "closed_at":   data.get("closed_at"),
                "comments":    data.get("comments", 0),
                "comments_url": data.get("comments_url"),
                "html_url":    data.get("html_url"),
                "pull_request": data.get("pull_request"),   # є якщо PR
            })

        _cache_set(cache_key, result)
        logger.info("Fetched issue %s/%s#%s", owner, repo, number)
        return result

    except httpx.TimeoutException:
        return json.dumps({"error": "Request timeout fetching issue"})
    except Exception as e:
        logger.error("Error fetching issue: %s", e)
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# Tool 2: search_similar_issues
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
async def search_similar_issues(repo: str, query: str, max_results: int = 5) -> str:
    """
    Search a GitHub repository for issues similar to the given query text.
    Uses GitHub search API. Results are cached.

    Args:
        repo:        Repository in 'owner/name' format, e.g. 'fastapi/fastapi'
        query:       Search keywords (title keywords, error messages, etc.)
        max_results: Maximum number of results to return (1-10)

    Returns:
        JSON array of matching issues with number, title, state, labels.
    """
    max_results = max(1, min(10, max_results))
    cache_key   = _make_key("search", f"{repo}:{query}:{max_results}")

    cached = _cache_get(cache_key)
    if cached:
        return cached

    search_query = f"{query} repo:{repo} is:issue"
    await asyncio.sleep(RATE_LIMIT_DELAY)

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.github.com/search/issues",
                headers=_github_headers(),
                params={"q": search_query, "per_page": max_results, "sort": "relevance"},
            )

        if resp.status_code == 422:
            return json.dumps({"error": "Invalid search query", "query": query})

        resp.raise_for_status()
        items = resp.json().get("items", [])

        result = json.dumps([
            {
                "number":     item.get("number"),
                "title":      item.get("title"),
                "state":      item.get("state"),
                "labels":     [l["name"] for l in item.get("labels", [])],
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "html_url":   item.get("html_url"),
                "score":      item.get("score"),
            }
            for item in items
        ])

        _cache_set(cache_key, result)
        logger.info("Search '%s' in %s → %d results", query[:40], repo, len(items))
        return result

    except Exception as e:
        logger.error("Search error: %s", e)
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# Tool 3: fetch_url (generic GitHub API fetcher)
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_url(url: str) -> str:
    """
    Fetch any GitHub API URL with authentication and caching.
    Use for: comments, repo file trees, PR data, user info.

    Args:
        url: GitHub API URL (must start with https://api.github.com/ or https://github.com/)

    Returns:
        JSON response from the API.
    """
    if not url.startswith(("https://api.github.com/", "https://github.com/")):
        return json.dumps({"error": "Only GitHub URLs are allowed"})

    cache_key = _make_key("url", url)
    cached    = _cache_get(cache_key)
    if cached:
        return cached

    await asyncio.sleep(RATE_LIMIT_DELAY)

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_github_headers())

        resp.raise_for_status()
        result = resp.text  # може бути JSON або HTML

        # Кешуємо тільки якщо не дуже великий
        if len(result) < 500_000:
            _cache_set(cache_key, result)

        logger.info("Fetched URL: %s (%d chars)", url[:60], len(result))
        return result

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {url}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# Tool 4: save_triage_note (local bookkeeping)
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
async def save_triage_note(issue_url: str, note: str) -> str:
    """
    Save a structured note about an issue for local bookkeeping and deduplication.
    Notes persist across agent runs for the same issue.

    Args:
        issue_url: The GitHub issue URL
        note:      The note content (classification, findings summary, etc.)

    Returns:
        Confirmation message.
    """
    try:
        db = _get_db()
        db.execute(
            "INSERT INTO notes (issue_url, note, created_at) VALUES (?, ?, ?)",
            (issue_url, note, time.time())
        )
        db.commit()
        logger.info("Note saved for %s", issue_url[:60])
        return json.dumps({"status": "saved", "issue_url": issue_url})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# Tool 5: get_triage_notes (local bookkeeping)
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_triage_notes(issue_url: str) -> str:
    """
    Retrieve previously saved triage notes for an issue.
    Useful for deduplication — check if the issue was already triaged.

    Args:
        issue_url: The GitHub issue URL

    Returns:
        JSON array of notes with content and timestamps.
    """
    try:
        db = _get_db()
        rows = db.execute(
            "SELECT note, created_at FROM notes WHERE issue_url = ? ORDER BY created_at DESC LIMIT 10",
            (issue_url,)
        ).fetchall()

        result = json.dumps([
            {"note": row[0], "created_at": row[1]}
            for row in rows
        ])
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting GitHub Triage MCP Server...")
    logger.info("Cache dir: %s", CACHE_DIR)
    logger.info("GitHub token: %s", "set" if GITHUB_TOKEN else "NOT SET (unauthenticated)")
    mcp.run()  # stdio transport за замовчуванням
