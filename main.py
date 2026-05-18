#!/usr/bin/env python3
"""
main.py — точка входу GitHub Triage Agent.

Використання:
    python main.py --url https://github.com/fastapi/fastapi/issues/1234
    python main.py --url URL --hint duplicate
    python main.py --eval        # запуск evaluation suite
"""
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import os
import sys
import uuid
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp_config import mcp_config_for_main
from agent.mcp_tools_context import set_mcp_tools
from agent.graph import build_graph
from agent.state import initial_state
from agent.trajectory_export import aggregate_token_usage, message_trace_from_state

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("main")

# ── LangSmith трасування (лише якщо явно увімкнено + є ключ) ────────
def _langsmith_enabled() -> bool:
    key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not key:
        return False
    flag = (os.getenv("LANGCHAIN_TRACING_V2") or "false").strip().lower()
    return flag in ("true", "1", "yes", "on")


if _langsmith_enabled():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "github-triage-agent")
    logger.info("LangSmith tracing enabled")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ── Основний runner ───────────────────────────────────────────────
async def run_triage(
    issue_url: str,
    task_hint: str | None = None,
    thread_id: str | None = None,
    human_in_loop: bool = True,
) -> dict:
    """
    Запускає агента для одного issue.
    Повертає dict з report, metadata, trajectory.
    """
    run_id    = str(uuid.uuid4())
    thread_id = thread_id or run_id
    config    = {"configurable": {"thread_id": thread_id}}

    logger.info("Starting triage: %s (run_id=%s)", issue_url, run_id)
    start_ts = datetime.now()

    # Два MCP-процеси stdio — див. таблицю в `agent/mcp_config.py` та `MCP_SERVERS_OVERVIEW`:
    #   1) github_triage → mcp_server/server.py (завжди)
    #   2) fetch         → FETCH_MCP_* з .env (mcp-server-fetch), опційно
    # langchain-mcp-adapters ≥0.1.0: MultiServerMCPClient is not an async context manager.
    mcp_client = MultiServerMCPClient(mcp_config_for_main())
    logger.info("Loading tools from MCP servers (can take 30–120s on cold start)...")
    mcp_tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in mcp_tools}
    logger.info("MCP tools loaded: %s", list(tools_by_name.keys()))

    config["configurable"]["tools"] = tools_by_name
    set_mcp_tools(tools_by_name)

    graph = build_graph()
    state = initial_state(issue_url=issue_url, task_hint=task_hint, run_id=run_id)

    result = await graph.ainvoke(state, config)

    snapshot = graph.get_state(config)
    is_interrupted = snapshot.next and "human_review" in snapshot.next

    if is_interrupted and human_in_loop:
        print("\n" + "═" * 60)
        print("  HUMAN REVIEW REQUESTED")
        print("═" * 60)
        print(f"\nIssue: {issue_url}")
        print("\nCurrent report:\n")
        print(result.get("triage_report", "No report yet"))
        print("\n" + "─" * 60)
        print("Type 'approve' to accept, or enter feedback to revise:")
        print("─" * 60)

        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            feedback = "approve"

        graph.update_state(config, {"human_feedback": feedback})
        set_mcp_tools(tools_by_name)
        result = await graph.ainvoke(None, config)

    duration_s = (datetime.now() - start_ts).total_seconds()

    msg_trace = message_trace_from_state(result)
    tok = aggregate_token_usage(msg_trace)

    trajectory = {
        "run_id":          run_id,
        "issue_url":       issue_url,
        "task_hint":       task_hint,
        "task_type":       result.get("task_type"),
        "tool_calls":      result.get("tool_calls_count", 0),
        "grounding_passed": result.get("grounding_passed"),
        "duration_s":      round(duration_s, 2),
        "error_count":     result.get("error_count", 0),
        "loop_count":      result.get("loop_count", 0),
        "tool_results":    result.get("tool_results", []),
        "message_trace":   msg_trace,
        "token_usage_est": tok,
        "timestamp":       start_ts.isoformat(),
    }

    return {
        "report":     result.get("triage_report", "No report generated"),
        "state":      result,
        "trajectory": trajectory,
    }


# ── CLI ───────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="GitHub Issue Triage Agent")
    parser.add_argument("--url",    help="GitHub issue URL")
    parser.add_argument("--hint",   help="Task hint: duplicate|code_area|stale|classify")
    parser.add_argument("--eval",   action="store_true", help="Run evaluation suite")
    parser.add_argument("--no-hil", action="store_true", help="Disable human-in-the-loop")
    parser.add_argument("--output", default="trajectories", help="Output dir for trajectories")
    parser.add_argument(
        "--list-mcp",
        action="store_true",
        help="Показати MCP-сервери й тулі з описами, вийти",
    )
    parser.add_argument(
        "--metrics-help",
        action="store_true",
        help="Пояснення метрик evaluation (друк у консоль, без запуску)",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Приклади issue URL з evaluation/tasks.json за типом (classify, duplicate, …)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Разом з --eval: після кожної задачі друкувати reasons/demerits і grounding",
    )
    args = parser.parse_args()

    if args.list_mcp:
        from agent.mcp_inventory import print_mcp_inventory

        await print_mcp_inventory()
        return

    if args.metrics_help:
        from evaluation.runner import print_metrics_help

        print_metrics_help()
        return

    if args.examples:
        from evaluation.runner import print_issue_type_examples

        print_issue_type_examples()
        return

    if args.eval:
        from evaluation.runner import run_evaluation
        await run_evaluation(output_dir=args.output, verbose=args.verbose)
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    result = await run_triage(
        issue_url=args.url,
        task_hint=args.hint,
        human_in_loop=not args.no_hil,
    )

    # Зберігаємо trajectory
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    traj_file = out_dir / f"{result['trajectory']['run_id']}.json"
    with open(traj_file, "w") as f:
        json.dump(result["trajectory"], f, indent=2, default=str)

    # Виводимо звіт
    print("\n" + "═" * 60)
    print("  TRIAGE REPORT")
    print("═" * 60)
    print(result["report"])
    print("\n" + "─" * 60)
    t = result["trajectory"]
    gp = t.get("grounding_passed")
    gp_str = "yes" if gp is True else ("no" if gp is False else "n/a")
    print(
        f"Task: {t['task_type']} | Tools: {t['tool_calls']} | "
        f"Grounding: {gp_str} | Time: {t['duration_s']}s"
    )
    print(f"Trajectory saved: {traj_file}")


if __name__ == "__main__":
    asyncio.run(main())
