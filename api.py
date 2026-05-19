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
import logging
from typing import Optional

from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from agent.launch_modes import resolve_launch, LAUNCH_MODES
from agent.prompt_quality import validate_user_prompt

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
    """
    Як CLI: `launch_mode` + `task_hint` + `user_prompt` проходять resolve_launch і валідацію промпта.
    Якщо `launch_mode` не передано — визначається з наявності hint/prompt (як у main.py).
    """
    issue_url: str
    launch_mode: Optional[str] = Field(
        default=None,
        description=f"Опціонально: {' | '.join(LAUNCH_MODES)}. None = авто з hint/prompt.",
    )
    task_hint: Optional[str] = Field(
        default=None,
        description="duplicate | code_area | stale | classify",
    )
    user_prompt: Optional[str] = Field(default=None, description="Інструкції користувача (укр/англ).")

class TriageResponse(BaseModel):
    job_id: str
    status: str
    message: str
    launch_mode: str = Field(description="Підтверджений режим після resolve_launch")
    resolved_task_hint: Optional[str] = Field(None, description="Hint, переданий у run_triage (після валідації)")
    prompt_warnings: Optional[list[str]] = Field(None, description="Попередження з validate_user_prompt, якщо були")

class TriageResult(BaseModel):
    job_id:             str
    status:             str
    issue_url:          str
    launch_mode:        Optional[str] = None
    task_type:          Optional[str]
    report:             Optional[str]
    grounding_passed:   Optional[bool]
    tool_calls:         Optional[int]
    duration_s:         Optional[float]
    fact_check:         Optional[dict] = None
    error:              Optional[str]
    created_at:         str


# ── Підготовка параметрів (як main.py) ────────────────────────────

def _prepare_triage_request(req: TriageRequest) -> tuple[str, Optional[str], Optional[str], Optional[list[str]]]:
    """
    Валідація промпта + resolve_launch.
    Повертає (launch_mode, task_hint, user_prompt, prompt_warnings).
    """
    user_prompt_raw = req.user_prompt
    prompt_warnings: Optional[list[str]] = None

    if user_prompt_raw is not None and str(user_prompt_raw).strip():
        v = validate_user_prompt(user_prompt_raw)
        if not v.is_valid:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_prompt", "issues": v.issues},
            )
        user_prompt_raw = v.normalized
        if v.warnings:
            prompt_warnings = v.warnings

    try:
        launch_mode, task_hint, user_prompt = resolve_launch(
            req.launch_mode,
            req.task_hint,
            user_prompt_raw,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "launch_mode", "message": str(e)},
        )

    return launch_mode, task_hint, user_prompt, prompt_warnings


# ── Background triage job ─────────────────────────────────────────

async def _run_triage_job(
    job_id: str,
    issue_url: str,
    launch_mode: str,
    task_hint: Optional[str],
    user_prompt: Optional[str] = None,
):
    """Запускає тріаж у фоні і зберігає результат."""
    _jobs[job_id]["status"] = "running"
    start = datetime.now()
    try:
        from main import run_triage
        result = await run_triage(
            issue_url=issue_url,
            task_hint=task_hint,
            user_prompt=user_prompt,
            thread_id=job_id,
            human_in_loop=False,   # API mode — no HIL
        )
        traj = result.get("trajectory") or {}
        traj["launch_mode"] = launch_mode
        duration = (datetime.now() - start).total_seconds()
        _jobs[job_id].update({
            "status":            "done",
            "launch_mode":       launch_mode,
            "report":            result.get("report"),
            "task_type":         traj.get("task_type"),
            "grounding_passed":  traj.get("grounding_passed"),
            "fact_check":        traj.get("fact_check"),
            "tool_calls":        traj.get("tool_calls"),
            "duration_s":        round(duration, 2),
            "error":             None,
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
    Перед постановкою в чергу: валідація `user_prompt` та `resolve_launch` (як у CLI).
    Повертає job_id для опитування статусу.
    """
    if not req.issue_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only GitHub issue URLs are supported")

    launch_mode, task_hint, user_prompt, prompt_warnings = _prepare_triage_request(req)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":           job_id,
        "status":           "queued",
        "issue_url":        req.issue_url,
        "launch_mode":      launch_mode,
        "task_hint":        task_hint,
        "user_prompt":      user_prompt,
        "prompt_warnings":  prompt_warnings,
        "created_at":       datetime.now().isoformat(),
        "report":           None,
        "error":            None,
        "fact_check":       None,
    }

    background_tasks.add_task(
        _run_triage_job, job_id, req.issue_url, launch_mode, task_hint, user_prompt
    )

    return TriageResponse(
        job_id=job_id,
        status="queued",
        message=f"Triage job started ({launch_mode}). Poll /triage/{job_id} for result.",
        launch_mode=launch_mode,
        resolved_task_hint=task_hint,
        prompt_warnings=prompt_warnings,
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
        launch_mode=job.get("launch_mode"),
        task_type=job.get("task_type"),
        report=job.get("report"),
        grounding_passed=job.get("grounding_passed"),
        tool_calls=job.get("tool_calls"),
        duration_s=job.get("duration_s"),
        fact_check=job.get("fact_check"),
        error=job.get("error"),
        created_at=job["created_at"],
    )


@app.get("/jobs")
async def list_jobs():
    """Список всіх задач (для дебагу)."""
    return {
        "total": len(_jobs),
        "jobs": [
            {
                "job_id":       j["job_id"],
                "status":       j["status"],
                "issue_url":    j["issue_url"],
                "launch_mode":  j.get("launch_mode"),
                "task_hint":    j.get("task_hint"),
            }
            for j in _jobs.values()
        ],
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
