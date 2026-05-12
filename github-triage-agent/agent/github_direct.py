"""Пряме завантаження issue з GitHub REST (fallback, якщо MCP-туля недоступна в вузлі)."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)")


def github_headers() -> dict[str, str]:
    token = (os.getenv("GITHUB_TOKEN") or "").strip()
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "github-triage-agent/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def compact_issue_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "number":       data.get("number"),
        "title":        data.get("title"),
        "body":         data.get("body"),
        "state":        data.get("state"),
        "labels":       [{"name": l["name"]} for l in data.get("labels", [])],
        "user":         {"login": data.get("user", {}).get("login")},
        "created_at":   data.get("created_at"),
        "updated_at":   data.get("updated_at"),
        "closed_at":    data.get("closed_at"),
        "comments":     data.get("comments", 0),
        "comments_url": data.get("comments_url"),
        "html_url":     data.get("html_url"),
        "pull_request": data.get("pull_request"),
        "_via":         "direct_http",
    }


async def fetch_github_issue_direct(url: str) -> dict[str, Any]:
    m = _URL_RE.match(url.strip())
    if not m:
        return {"error": f"Invalid GitHub issue URL: {url}"}
    owner, repo, num = m.group(1), m.group(2), m.group(3)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
    # GitHub перенаправляє деякі repos (301); без follow_redirects json може бути не issue.
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(api_url, headers=github_headers())
    if resp.status_code == 404:
        return {"message": "Not Found", "url": url}
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status_code": resp.status_code, "body": resp.text[:500]}
    raw = resp.json()
    out = compact_issue_payload(raw)
    _logger.info(
        "direct fetch issue#%s HTTP %s — body_len=%d",
        num,
        resp.status_code,
        len(out.get("body") or ""),
    )
    return out
