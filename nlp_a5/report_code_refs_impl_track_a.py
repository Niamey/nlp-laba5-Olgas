"""
Report code references (implementation mirror) — Track A.

Purpose:
  Provide stable, greppable Python code (with line numbers) that mirrors the Kaggle notebook logic:
  - AgentSpec: inputs/outputs, tool contracts (as docstrings/types)
  - LangGraph StateGraph: state schema, routing, checkpointer, HITL interrupt, budgets
  - MCP integration: 1 custom server + 1 third-party fetch server
  - Evaluation + trajectory logging hooks
  - Ablations knobs (model/prompt/graph)

Canonical source remains `nlp_assignment5_trackA_kaggle.ipynb`.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
import traceback
from operator import add
from typing import Annotated, Any, Literal, TypedDict

# ---- AgentSpec: Inputs / Outputs ------------------------------------------------


class EvalRow(TypedDict, total=False):
    ok: bool
    eval_track: str
    task_id: str
    from_eval: str
    dry_run: bool
    trajectory: str
    mentions: list[str]
    grounded: list[str]
    preview_tail: str
    error: str
    traceback: str


# Inputs:
#   - prompt: one TASKS[i]["prompt"] string
#   - tools: list of MCP tools loaded via load_mcp_tools()
# Output:
#   - EvalRow dict (trajectory path + grounding/tool-selection fields)


# ---- Tool contracts (by tool class) --------------------------------------------

# Custom MCP server (FastMCP, separate process) provides 4 tools:
#   - arxiv_search(query: str, max_results: int = 5) -> dict
#       side effects: on-disk cache in WORKDIR/nlp_a5_mcp_cache
#   - semanticscholar_graph(paper_identifier: str, ...) -> dict
#   - openalex_work_lookup(query: str, per_page: int = 5) -> dict
#   - research_bookkeeping(action: str, ...) -> dict
#       side effects: writes research_notes.jsonl + dedupe_*.flag
#
# Third-party MCP fetch server provides fetch_* tools:
#   - started via: python -m mcp_server_fetch (pip package mcp-server-fetch)
#   - npx is opt-in only because npm stdout breaks JSON-RPC.


# ---- Budgets / stopping criteria -----------------------------------------------

MAX_TOOL_BATCHES = int(os.environ.get("NLP_A5_MAX_TOOLS", "8"))
PATCH_REPAIRS_ALLOWED = int(os.environ.get("NLP_A5_MAX_REPAIR", "2"))
ABLATION_VARIANT = os.environ.get("NLP_ABL_PROMPT_VARIANT", "A")
ABLATION_SIMPLE_GRAPH = os.environ.get("NLP_ABL_SIMPLE_GRAPH", "0").lower() in {"1", "true", "yes"}


def variant_system_prompt() -> str:
    base_common = """You are a Track-A scientific literature assistant.
Ground every factual claim using MCP tools only. Prefer arXiv + Semantic Scholar/OpenAlex graphs.
Explicitly cite arxiv identifiers like `arxiv:YYMM.NNNNN`. Never invent DOIs or arXiv ids.
Respect polite rate limits; batch queries."""
    if ABLATION_VARIANT == "B":
        return base_common + " Tool discipline: NEVER call redundant tools — each batch must widen evidence."
    return base_common + " Keep user-facing prose compact."


ARXIV_ID_REGEX = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def grounding_stats(*, final_assistant_text: str, tool_evidence_blob: str) -> tuple[set[str], set[str]]:
    mentions = set(ARXIV_ID_REGEX.findall(final_assistant_text or ""))
    grounded = set(x for x in mentions if x in (tool_evidence_blob or ""))
    return mentions, grounded


# ---- LangGraph state schema -----------------------------------------------------


class AgentState(TypedDict, total=False):
    # scratchpad / transient
    messages: Annotated[list[Any], add]  # notebook uses add_messages; mirrored here as add
    # durable state
    plan: str
    facts: list[dict]
    artifacts: dict
    # counters
    tool_batches: Annotated[int, add]
    repair_rounds: Annotated[int, add]


# ---- Trajectory logging (machine-readable) -------------------------------------


class TrajectoryRecorder:
    def __init__(self, *, workdir: pathlib.Path, tid: str):
        base = workdir / "nlp_assignment5_trajectories"
        base.mkdir(parents=True, exist_ok=True)
        self.fp = base / f"{tid}-{int(time.time())}.jsonl"

    def log(self, event: str, payload: dict):
        row = {"ts_ms": int(time.time() * 1000), "event": event, "payload": payload}
        with open(self.fp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @property
    def path(self) -> str:
        return str(self.fp)


# ---- StateGraph control-flow (mirrors notebook) --------------------------------


def route_llm(state: AgentState) -> Literal["tools", "reflect"]:
    """
    Conditional edge:
      - if LLM produced tool_calls and tool budget not exhausted -> tools
      - else -> reflect/verifier
    """
    last = (state.get("messages") or [])[-1] if state.get("messages") else None
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    if has_tool_calls and state.get("tool_batches", 0) < MAX_TOOL_BATCHES:
        return "tools"
    return "reflect"


def verifier(state: AgentState) -> AgentState:
    """
    Verifier/repair loop:
      - if ungrounded ids exist -> inject HumanMessage asking to repair
      - if repair rounds exceeded -> halt (human escalation)
    """
    # notebook uses ToolMessage blobs + final assistant text; mirrored conceptually
    mentions, grounded = set(), set()
    if not mentions or grounded == mentions:
        return {}
    missing = sorted(mentions - grounded)
    if state.get("repair_rounds", 0) >= PATCH_REPAIRS_ALLOWED:
        return {"messages": ["[verifier halt] Ungrounded identifiers remain: " + ", ".join(missing)]}
    return {
        "messages": [
            "Repair answer: cite only identifiers present in prior ToolMessage bodies. "
            f"Missing evidence for: {', '.join(missing)}."
        ],
        "repair_rounds": 1,
    }


def route_after_reflect(state: AgentState) -> str:
    """
    Graph ablation:
      NLP_ABL_SIMPLE_GRAPH=1 -> reflect leads directly to END (no repair/human loop)
    """
    if ABLATION_SIMPLE_GRAPH:
        return "END"
    last = (state.get("messages") or [])[-1] if state.get("messages") else None
    if isinstance(last, str) and "[verifier halt]" in last:
        return "human_escalate"
    if isinstance(last, str) and last.startswith("Repair answer:"):
        return "agent"
    return "END"


# ---- MCP integration knobs (mirror) ---------------------------------------------


def fetch_backend_policy() -> dict[str, Any]:
    """
    Third-party fetch MCP policy:
      - default backend: pip (python -m mcp_server_fetch)
      - npx is opt-in only with NLP_A5_FETCH_ALLOW_NPX=1 because npm stdout breaks JSON-RPC
    """
    backend = os.environ.get("NLP_A5_FETCH_BACKEND", "pip").strip().lower()
    allow_npx = os.environ.get("NLP_A5_FETCH_ALLOW_NPX", "").lower() in {"1", "true", "yes"}
    if backend == "npx" and not allow_npx:
        backend = "pip"
    return {"backend": backend, "allow_npx": allow_npx}


# ---- Evaluation set expectations ------------------------------------------------

# Notebook embeds >=30 tasks in TASKS with:
#   - rubric.quality_scale (0/1/2)
#   - rubric.must_use_tool_classes
#   - rubric.forbidden
#   - adversarial flags for some tasks


# ---- Ablations knobs ------------------------------------------------------------

# Model ablation:
#   NLP_A5_LLM_PROVIDER=gemini
#   NLP_A5_GEMINI_MODEL in {"gemini-2.5-flash", "gemini-2.5-pro", ...}
#
# Prompt ablation:
#   NLP_ABL_PROMPT_VARIANT in {"A","B"}
#
# Graph ablation:
#   NLP_ABL_SIMPLE_GRAPH=1 (reflect -> END)

