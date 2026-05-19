#!/usr/bin/env python3
"""
main.py — точка входу GitHub Triage Agent.

Використання:
    python main.py --url https://github.com/fastapi/fastapi/issues/1234
    # Шаблони для тестів (hint+prompt та лише prompt): prompts/TESTING_EXAMPLES.md
    python main.py --launch-examples

    # Чотири режими (--launch-mode; без нього режим виводиться з --hint/--prompt):
    python main.py --launch-mode auto         --url URL --no-hil
    python main.py --launch-mode hint-only    --hint duplicate --url URL --no-hil
    python main.py --launch-mode hint-prompt  --hint classify  --url URL --prompt-file prompts/classify.txt --no-hil
    python main.py --launch-mode prompt-auto  --url URL --no-hil --prompt "Знайди схожі issues, чи це дублікат"

    python main.py --eval
    python main.py --eval --prompt "Зроби мінімум 2 пошуки дублікатів і поясни висновок"
    python main.py --eval --prompt-file prompts/duplicate.txt --tasks T002 T011
    # Після eval додатково: два блоки «БЛОК EVAL» (зведення + по кожній задачі) у консолі.

    # Пояснення метрик на одному прикладі (після одного URL):
    python main.py --url URL --no-hil --explain-metrics
    python main.py --eval --print-metrics

    # Усі метрики в один файл + звіт у консолі:
    python main.py --url URL --no-hil --metrics-out metrics.json
    python main.py --eval --metrics-out eval_metrics.json
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
def _load_user_prompt(cli_prompt: str | None, prompt_file: str | None) -> str | None:
    from agent.user_instructions import load_user_prompt_from_cli

    return load_user_prompt_from_cli(cli_prompt, prompt_file)


def _strip_message_trace_for_print(obj: object) -> object:
    """Прибирає message_trace з вкладених dict (для компактного друку метрик)."""
    if isinstance(obj, dict):
        return {
            k: _strip_message_trace_for_print(v)
            for k, v in obj.items()
            if k != "message_trace"
        }
    if isinstance(obj, list):
        return [_strip_message_trace_for_print(x) for x in obj]
    return obj


_TRAJECTORY_METRIC_HELP_UK: dict[str, str] = {
    "run_id": "Ідентифікатор прогону (UUID).",
    "issue_url": "Issue на GitHub, який тріажили.",
    "launch_mode": "Режим запуску CLI (auto / hint-only / …) після resolve_launch.",
    "task_hint": "Підказка гілки графа (duplicate, classify, …) або null.",
    "user_prompt": "Додаткові інструкції користувача або null.",
    "task_type": "Тип тріажу, визначений агентом (результат роботи).",
    "tool_calls": "Кількість викликів MCP-інструментів за прогін.",
    "grounding_passed": "Чи пройшла перевірка «звіт спирається на дані з тулів» (true/false).",
    "fact_check": "Regex-перевірка фактів у звіті (#issues, @user, файли, URL) проти контенту.",
    "duration_s": "Тривалість прогону в секундах.",
    "error_count": "Лічильник помилок під час виконання графа.",
    "loop_count": "Скільки разів пройшли цикл до фінального звіту.",
    "tool_results": "Сирі результати викликів інструментів (доказова база).",
    "message_trace": "Журнал повідомлень LLM (для оцінки токенів і дебагу).",
    "token_usage_est": "Оцінка input/output токенів за trace.",
    "timestamp": "Початок прогону (ISO).",
    "report": "Текст підсумкового triage-звіту.",
}


def _preview_metric_value(key: str, val: object) -> str:
    """Коротке значення для показу в «один приклад»."""
    if val is None:
        return "—"
    if key == "report" and isinstance(val, str):
        s = val.strip().replace("\n", " ")[:140]
        return f"{len(val)} симв.; приклад: {s!r}…"
    if key == "user_prompt" and isinstance(val, str):
        return f"{len(val)} симв." if val else "—"
    if key == "tool_results" and isinstance(val, list):
        tools = [x.get("tool") for x in val if isinstance(x, dict)]
        return f"{len(val)} викликів" + (f": {tools[:8]}{'…' if len(tools) > 8 else ''}" if tools else "")
    if key == "message_trace" and isinstance(val, list):
        return f"{len(val)} записів"
    if key == "fact_check" and isinstance(val, dict):
        return (
            f"passed={val.get('passed')}, "
            f"rate={val.get('facts_grounded_rate')}, "
            f"facts={val.get('grounded_facts')}/{val.get('total_facts')}"
        )
    if key == "token_usage_est" and isinstance(val, dict):
        return (
            f"in≈{val.get('input_tokens_est')}, out≈{val.get('output_tokens_est')}, "
            f"total≈{val.get('total_tokens_est')}"
        )
    if isinstance(val, str) and len(val) > 90:
        return repr(val[:90]) + "…"
    return repr(val)


def print_trajectory_metrics_explained(export: dict) -> None:
    """Друкує список зібраних метрик з поясненнями на одному прикладі прогону."""
    print("\n" + "═" * 60)
    print("  МЕТРИКИ (один приклад прогону) — що зібрано і яке значення")
    print("═" * 60)
    for key in sorted(export.keys()):
        desc = _TRAJECTORY_METRIC_HELP_UK.get(key, "Поле з траєкторії / звіту.")
        val_line = _preview_metric_value(key, export.get(key))
        print(f"\n  • {key}\n    {desc}\n    Значення: {val_line}")
    print("\n" + "═" * 60 + "\n")


async def run_triage(
    issue_url: str,
    task_hint: str | None = None,
    user_prompt: str | None = None,
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

    # langchain-mcp-adapters ≥0.1.0: MultiServerMCPClient is not an async context manager.
    mcp_client = MultiServerMCPClient(mcp_config_for_main())
    logger.info("Loading tools from MCP servers (can take 30–120s on cold start)...")
    mcp_tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in mcp_tools}
    logger.info("MCP tools loaded: %s", list(tools_by_name.keys()))

    config["configurable"]["tools"] = tools_by_name
    set_mcp_tools(tools_by_name)

    graph = build_graph()
    state = initial_state(
        issue_url=issue_url,
        task_hint=task_hint,
        user_prompt=user_prompt,
        run_id=run_id,
    )

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
        "launch_mode":     None,  # заповнюється в main() при CLI
        "task_hint":       task_hint,
        "user_prompt":     user_prompt,
        "task_type":       result.get("task_type"),
        "tool_calls":      result.get("tool_calls_count", 0),
        "grounding_passed": result.get("grounding_passed"),
        "fact_check":      result.get("fact_check"),
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
    parser.add_argument(
        "--hint",
        help="Режим тріажу: duplicate | code_area | stale | classify (фіксує гілку графа)",
    )
    parser.add_argument(
        "--prompt",
        help="Довільні інструкції для цього прогону (додаються до промптів планувальника, воркера, звіту)",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="Файл з інструкціями (UTF-8); має пріоритет над --prompt",
    )
    parser.add_argument("--eval",   action="store_true", help="Run evaluation suite")
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="T00x",
        help="Разом з --eval: запустити лише ці task_id з evaluation/tasks.json",
    )
    parser.add_argument("--no-hil", action="store_true", help="Disable human-in-the-loop")
    parser.add_argument("--output", default="trajectories", help="Output dir for trajectories")
    parser.add_argument(
        "--list-mcp",
        action="store_true",
        help="Показати MCP-сервери й тулі з описами, вийти",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["auto", "hint-only", "hint-prompt", "prompt-auto"],
        help=(
            "auto: без hint/prompt | hint-only: лише --hint | "
            "hint-prompt: --hint + --prompt | prompt-auto: лише --prompt (тип обере планувальник)"
        ),
    )
    parser.add_argument(
        "--launch-examples",
        action="store_true",
        help="Показати 3 режими запуску з прикладами команд",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Після тріажу вивести в stdout усі зібрані метрики (JSON trajectory + report). За замовчуванням без message_trace.",
    )
    parser.add_argument(
        "--print-metrics-full",
        action="store_true",
        help="Разом з --print-metrics: включити message_trace (дуже багато тексту).",
    )
    parser.add_argument(
        "--metrics-out",
        metavar="PATH",
        help="Зберегти всі зібрані метрики в один JSON-файл (одиночний прогон або повний eval summary).",
    )
    parser.add_argument(
        "--metrics-out-full",
        action="store_true",
        help="Разом з --metrics-out: включити message_trace у результат (дуже великий файл).",
    )
    parser.add_argument(
        "--explain-metrics",
        action="store_true",
        help="Після тріажу на одному issue показати список полів-метрик з поясненнями та значеннями з цього прогону.",
    )
    parser.add_argument(
        "--extended-metrics",
        action="store_true",
        help="Після тріажу: блок розширених метрик (якість звіту, search, tools, галюцинації, вартість). "
        "Також пишуться в trajectory JSON як extended_metrics.",
    )
    parser.add_argument(
        "--task-id",
        metavar="T00x",
        help="Для одного URL: підставити критерії з evaluation/tasks.json (напр. T002) у rubric_autocheck.",
    )
    args = parser.parse_args()

    if args.list_mcp:
        from agent.mcp_inventory import print_mcp_inventory

        await print_mcp_inventory()
        return

    if args.launch_examples:
        from agent.launch_modes import print_launch_examples

        print_launch_examples(args.url or "https://github.com/owner/repo/issues/123")
        return

    if args.eval:
        from evaluation.runner import run_evaluation

        eval_user_prompt = _load_user_prompt(args.prompt, args.prompt_file)
        if eval_user_prompt:
            from agent.prompt_quality import validate_user_prompt, format_validation_report

            ev_val = validate_user_prompt(eval_user_prompt)
            if not ev_val.is_valid:
                print("Помилка промпта (eval):", file=sys.stderr)
                for issue in ev_val.issues:
                    print(f"  ✗ {issue}", file=sys.stderr)
                sys.exit(2)
            eval_user_prompt = ev_val.normalized
            print(f"\n  Eval user prompt: {len(eval_user_prompt)} chars")
            report = format_validation_report(ev_val)
            if report:
                print(report)

        await run_evaluation(
            output_dir=args.output,
            tasks_filter=args.tasks,
            user_prompt=eval_user_prompt,
        )
        summary_path = Path(args.output) / "summary.json"
        if args.metrics_out:
            if summary_path.is_file():
                ev = json.loads(summary_path.read_text(encoding="utf-8"))
                payload = ev if args.metrics_out_full else _strip_message_trace_for_print(ev)
                mo = Path(args.metrics_out)
                mo.parent.mkdir(parents=True, exist_ok=True)
                mo.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                print(f"Усі метрики eval: {mo.resolve()}")
            else:
                print(f"Немає {summary_path} — metrics-out пропущено.", file=sys.stderr)
        if args.print_metrics:
            if summary_path.is_file():
                ev = json.loads(summary_path.read_text(encoding="utf-8"))
                if not args.print_metrics_full:
                    ev = _strip_message_trace_for_print(ev)
                print("\n" + "═" * 60)
                print(f"  METRICS — eval summary ({summary_path})")
                print("═" * 60)
                print(json.dumps(ev, indent=2, ensure_ascii=False, default=str))
            else:
                print(f"Немає {summary_path} для друку метрик.", file=sys.stderr)
        if args.explain_metrics:
            print(
                "\n  --explain-metrics діє для одного issue (--url). "
                "Після eval повні зведення: trajectories/summary.json та "
                "python scripts/show_eval_summary.py --all --per-task\n"
            )
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    from agent.launch_modes import resolve_launch
    from agent.prompt_quality import validate_user_prompt, format_validation_report

    user_prompt_raw = _load_user_prompt(args.prompt, args.prompt_file)

    # Валідація промпта (лише якщо щось передали; для auto без промпта це no-op)
    validation = None
    if user_prompt_raw is not None and str(user_prompt_raw).strip():
        validation = validate_user_prompt(user_prompt_raw)
        if not validation.is_valid:
            print("Помилка промпта:", file=sys.stderr)
            for issue in validation.issues:
                print(f"  ✗ {issue}", file=sys.stderr)
            print(
                "\n  Приклади валідних промптів:\n"
                "    --prompt \"Знайди схожі issues, чи це дублікат?\"\n"
                "    --prompt \"Який модуль/файл зачеплений?\"\n"
                "    --prompt \"Це bug чи feature? Запропонуй labels.\"\n"
                "    --prompt-file prompts/duplicate.txt",
                file=sys.stderr,
            )
            sys.exit(2)
        user_prompt_raw = validation.normalized

    try:
        launch_mode, task_hint, user_prompt = resolve_launch(
            args.launch_mode,
            args.hint,
            user_prompt_raw,
        )
    except ValueError as e:
        print(f"Помилка режиму запуску: {e}", file=sys.stderr)
        print("  Підказка: python main.py --launch-examples", file=sys.stderr)
        sys.exit(2)

    print(f"\n  Launch mode: {launch_mode}")
    if task_hint:
        print(f"  Hint:        {task_hint}")
    if user_prompt:
        print(f"  User prompt: {len(user_prompt)} chars")
        if validation is not None:
            report = format_validation_report(validation)
            if report:
                print(report)

    result = await run_triage(
        issue_url=args.url,
        task_hint=task_hint,
        user_prompt=user_prompt,
        human_in_loop=not args.no_hil,
    )
    result["trajectory"]["launch_mode"] = launch_mode

    eval_task = None
    if args.task_id:
        tasks_path = Path(__file__).resolve().parent / "evaluation" / "tasks.json"
        if tasks_path.is_file():
            import json as _json

            for t in _json.loads(tasks_path.read_text(encoding="utf-8")):
                if t.get("task_id") == args.task_id.strip():
                    eval_task = t
                    break
        if eval_task is None:
            print(f"Попередження: task_id {args.task_id!r} не знайдено в tasks.json", file=sys.stderr)

    from agent.extended_metrics import compute_extended_metrics, print_extended_metrics

    extended = compute_extended_metrics(
        report=result.get("report", ""),
        state=result.get("state") or {},
        trajectory=result["trajectory"],
        task=eval_task,
    )
    result["trajectory"]["extended_metrics"] = extended

    # Зберігаємо trajectory
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    traj_file = out_dir / f"{result['trajectory']['run_id']}.json"
    traj_export = dict(result["trajectory"])
    traj_export["report"] = result.get("report")
    traj_export["extended_metrics"] = extended
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump(traj_export, f, indent=2, ensure_ascii=False, default=str)

    # Виводимо звіт
    print("\n" + "═" * 60)
    print("  TRIAGE REPORT")
    print("═" * 60)
    print(result["report"])
    print("\n" + "─" * 60)
    t = result["trajectory"]
    print(f"Task: {t['task_type']} | Tools: {t['tool_calls']} | Time: {t['duration_s']}s")
    print(f"Trajectory saved: {traj_file}")

    if args.metrics_out:
        payload = traj_export if args.metrics_out_full else _strip_message_trace_for_print(traj_export)
        mo = Path(args.metrics_out)
        mo.parent.mkdir(parents=True, exist_ok=True)
        mo.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Усі метрики в одному файлі: {mo.resolve()}")

    if args.print_metrics:
        payload = traj_export if args.print_metrics_full else _strip_message_trace_for_print(traj_export)
        print("\n" + "═" * 60)
        print("  METRICS (JSON — те саме, що в trajectory JSON + report)")
        print("═" * 60)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    if args.explain_metrics:
        print_trajectory_metrics_explained(traj_export)

    if args.extended_metrics:
        print_extended_metrics(extended, issue_url=args.url or "")


if __name__ == "__main__":
    asyncio.run(main())
