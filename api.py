"""
api.py — FastAPI HTTP інтерфейс для агента.
Дозволяє запускати тріаж через REST API (для деплою).

Запуск:
    uvicorn api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import os
import uuid
import asyncio
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("api")

app = FastAPI(
    title="GitHub Issue Triage Agent",
    description="LangGraph + MCP agent for automated GitHub issue triage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (для prod використати Redis)
_jobs: dict[str, dict] = {}


# ── Schemas ───────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    issue_url: str
    task_hint: Optional[str] = None

class TriageResponse(BaseModel):
    job_id:    str
    status:    str
    message:   str

class TriageResult(BaseModel):
    job_id:         str
    status:         str
    issue_url:      str
    task_type:      Optional[str]
    report:         Optional[str]
    grounding_passed: Optional[bool]
    tool_calls:     Optional[int]
    duration_s:     Optional[float]
    error:          Optional[str]
    created_at:     str


# ── Background triage job ─────────────────────────────────────────

async def _run_triage_job(job_id: str, issue_url: str, task_hint: Optional[str]):
    """Запускає тріаж у фоні і зберігає результат."""
    _jobs[job_id]["status"] = "running"
    start = datetime.now()
    try:
        from main import run_triage
        result = await run_triage(
            issue_url=issue_url,
            task_hint=task_hint,
            thread_id=job_id,
            human_in_loop=False,   # API mode — no HIL
        )
        duration = (datetime.now() - start).total_seconds()
        traj = result.get("trajectory", {})
        _jobs[job_id].update({
            "status":          "done",
            "report":          result.get("report"),
            "task_type":       traj.get("task_type"),
            "grounding_passed": traj.get("grounding_passed"),
            "tool_calls":      traj.get("tool_calls"),
            "duration_s":      round(duration, 2),
            "error":           None,
        })
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e)
        _jobs[job_id].update({
            "status": "error",
            "error":  str(e),
        })


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest, background_tasks: BackgroundTasks):
    """
    Запускає тріаж issue асинхронно.
    Повертає job_id для опитування статусу.
    """
    if not req.issue_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only GitHub issue URLs are supported")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "issue_url":  req.issue_url,
        "task_hint":  req.task_hint,
        "created_at": datetime.now().isoformat(),
        "report":     None,
        "error":      None,
    }

    background_tasks.add_task(
        _run_triage_job, job_id, req.issue_url, req.task_hint
    )

    return TriageResponse(
        job_id=job_id,
        status="queued",
        message=f"Triage job started. Poll /triage/{job_id} for result.",
    )


@app.get("/triage/{job_id}", response_model=TriageResult)
async def get_triage(job_id: str):
    """Отримує статус і результат тріажу."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return TriageResult(
        job_id=job_id,
        status=job["status"],
        issue_url=job["issue_url"],
        task_type=job.get("task_type"),
        report=job.get("report"),
        grounding_passed=job.get("grounding_passed"),
        tool_calls=job.get("tool_calls"),
        duration_s=job.get("duration_s"),
        error=job.get("error"),
        created_at=job["created_at"],
    )


@app.get("/jobs")
async def list_jobs():
    """Список всіх задач (для дебагу)."""
    return {
        "total": len(_jobs),
        "jobs": [
            {"job_id": j["job_id"], "status": j["status"], "issue_url": j["issue_url"]}
            for j in _jobs.values()
        ]
    }


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
