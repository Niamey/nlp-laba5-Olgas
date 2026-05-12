#!/usr/bin/env python3
"""
Custom MCP server for NLP Assignment 5 — Track A (literature).
Run as a separate process:  python literature_mcp_server.py
Requires: pip install mcp httpx  (httpx optional; stdlib urllib used by default)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

IS_KAGGLE = os.path.exists("/kaggle/working")
_DEFAULT_CACHE = Path("/kaggle/working/nlp_a5_mcp_cache" if IS_KAGGLE else Path(__file__).resolve().parent / "mcp_cache")
CACHE_DIR = Path(os.environ.get("NLP_A5_MCP_CACHE", str(_DEFAULT_CACHE)))
NOTES_PATH = CACHE_DIR / "research_notes.jsonl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(name="nlp-a5-literature")


def _cache_key(kind: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"k": kind, "p": payload}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def _cache_get(kind: str, payload: dict[str, Any]) -> Any | None:
    ck = _cache_key(kind, payload)
    path = CACHE_DIR / f"{ck}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)["value"]
    return None


def _cache_set(kind: str, payload: dict[str, Any], value: Any) -> None:
    ck = _cache_key(kind, payload)
    path = CACHE_DIR / f"{ck}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kind": kind, "payload": payload, "value": value}, f, ensure_ascii=False)


def _http_json(url: str, timeout_s: float = 30.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NLP_Assignment5_TrackA_EDU/1.0 (contact: student; polite pool)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_text(url: str, timeout_s: float = 30.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NLP_Assignment5_TrackA_EDU/1.0 (contact: student; polite pool)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _arxiv_parse_atom(xml_text: str) -> list[dict[str, Any]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for ent in root.findall("a:entry", ns):
        id_url = (ent.findtext("a:id", default="", namespaces=ns) or "").strip()
        m = re.search(r"arxiv\.org/abs/([^?#]+)", id_url)
        arxiv_id = m.group(1) if m else id_url
        title = re.sub(r"\s+", " ", (ent.findtext("a:title", default="", namespaces=ns) or "")).strip()
        summary = re.sub(r"\s+", " ", (ent.findtext("a:summary", default="", namespaces=ns) or "")).strip()
        updated = (ent.findtext("a:updated", default="", namespaces=ns) or "").strip()
        authors = []
        for a in ent.findall("a:author", ns):
            name = (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary[:6000],
                "updated": updated,
                "authors": authors[:20],
                "arxiv_abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return out


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search arXiv Atom API for papers. Returns structured hits (ids, titles, abstracts, URLs). Rate-limited via cache."""
    max_results = int(max(min(max_results, 30), 1))
    qp = urllib.parse.quote_plus(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{qp}&start=0&max_results={max_results}"
    payload = {"q": query, "n": max_results}
    cached = _cache_get("arxiv_search", payload)
    if cached is not None:
        return {"cached": True, **cached}
    _rate_sleep = float(os.environ.get("NLP_A5_ARXIV_SLEEP", "3.5"))
    time.sleep(min(_rate_sleep, 5.0))
    xml_text = _http_text(url, timeout_s=45.0)
    hits = _arxiv_parse_atom(xml_text)
    body = {"query": query, "max_results": max_results, "hits": hits, "arxiv_api_url": url}
    _cache_set("arxiv_search", payload, body)
    return {"cached": False, **body}


def _s2_paper_id_from_arxiv(arxiv_id: str) -> str:
    ax = arxiv_id.strip().replace("arXiv:", "").replace("arxiv:", "")
    return f"ARXIV:{ax}"


@mcp.tool()
def semanticscholar_graph(
    paper_identifier: str,
    include_citations: bool = True,
    include_references: bool = True,
    limit_citations: int = 20,
    limit_references: int = 20,
) -> dict[str, Any]:
    """Fetch Semantic Scholar metadata, citation list, and reference list for a paper id (arXiv id, DOI, or S2 id)."""
    limit_citations = int(max(min(limit_citations, 100), 0))
    limit_references = int(max(min(limit_references, 100), 0))
    pid = paper_identifier.strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", pid):
        pid = _s2_paper_id_from_arxiv(pid)
    field_parts = ["title", "abstract", "year", "authors", "externalIds", "url", "isOpenAccess", "citationCount", "referenceCount"]
    if include_citations:
        field_parts.append("citations.paperId,citations.title,citations.year,citations.externalIds")
    if include_references:
        field_parts.append("references.paperId,references.title,references.year,references.externalIds")

    qs = urllib.parse.quote(",".join(field_parts), safe=",")
    enc_id = urllib.parse.quote(pid, safe="")
    api = f"https://api.semanticscholar.org/graph/v1/paper/{enc_id}?fields={qs}"
    payload = {
        "paper_identifier": paper_identifier.strip(),
        "include_citations": include_citations,
        "include_references": include_references,
        "limit_citations": limit_citations,
        "limit_references": limit_references,
    }
    cached = _cache_get("s2_graph", payload)
    if cached is not None:
        return {"cached": True, **cached}
    time.sleep(float(os.environ.get("NLP_A5_S2_SLEEP", "1.25")))
    data = _http_json(api, timeout_s=45.0)
    if include_citations and isinstance(data.get("citations"), list):
        data["citations"] = data["citations"][:limit_citations]
    if include_references and isinstance(data.get("references"), list):
        data["references"] = data["references"][:limit_references]
    slim = {"paper": data, "requested_url": api}
    _cache_set("s2_graph", payload, slim)
    return {"cached": False, **slim}


@mcp.tool()
def openalex_work_lookup(query: str, per_page: int = 5) -> dict[str, Any]:
    """Search OpenAlex works (titles, IDs, venues, OA status). Supplementary bibliographic grounding."""
    per_page = int(max(min(per_page, 25), 1))
    qp = urllib.parse.quote_plus(query)
    url = f"https://api.openalex.org/works?search={qp}&per_page={per_page}"
    payload = {"q": query, "n": per_page}
    cached = _cache_get("openalex_search", payload)
    if cached is not None:
        return {"cached": True, **cached}
    data = _http_json(url)
    results = []
    for w in (data.get("results") or [])[:per_page]:
        doi_raw = w.get("doi") or ""
        if isinstance(doi_raw, str) and doi_raw.startswith("https://doi.org/"):
            doi_raw = doi_raw.split("https://doi.org/", 1)[-1]
        results.append(
            {
                "openalex_id": w.get("id"),
                "title": w.get("display_name"),
                "publication_year": w.get("publication_year"),
                "doi": doi_raw,
            }
        )
    body = {"query": query, "openalex_search_url": url, "hits": results}
    _cache_set("openalex_search", payload, body)
    return {"cached": False, **body}


@mcp.tool()
def research_bookkeeping(
    action: str,
    task_id: str = "",
    content: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    """Structured local bookkeeping: append notes JSONL or record dedupe keys inside the on-disk cache (no secrets)."""
    action = action.strip().lower()
    content = content or {}
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if action == "append_note":
        rec = {"task_id": task_id, "content": content, "ts_ms": int(time.time() * 1000)}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(NOTES_PATH, "a", encoding="utf-8") as f:
            f.write(line)
        return {"ok": True, "records_path": str(NOTES_PATH), "record_preview": rec}
    if action == "stats":
        stats = sorted([p.name for p in CACHE_DIR.glob("*.json")])
        n_cache = len(stats)
        notes_n = NOTES_PATH.stat().st_size // 80 if NOTES_PATH.exists() else 0
        return {"ok": True, "cache_files": n_cache, "research_notes_est_lines": notes_n, "notes_path": str(NOTES_PATH)}
    if action == "dedupe_seen":
        if not dedupe_key:
            raise ValueError("dedupe_key required")
        dk = dedupe_key.strip()
        dk_hash = hashlib.sha256(dk.encode("utf-8")).hexdigest()[:48]
        mark = CACHE_DIR / f"dedupe_{dk_hash}.flag"
        if mark.exists():
            return {"ok": True, "seen": True, "dedupe_key": dk}
        mark.write_text(json.dumps({"key": dk, "ts_ms": int(time.time() * 1000)}), encoding="utf-8")
        return {"ok": True, "seen": False, "dedupe_key": dk}
    raise ValueError("action must be one of: append_note, stats, dedupe_seen")


if __name__ == "__main__":
    mcp.run()
