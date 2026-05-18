# Демо та метрики (захист / доповідь)

## 1. Два типи MCP

| Тип | Ім’я в клієнті | Призначення |
|-----|----------------|------------|
| **Власний** | `github_triage` | FastMCP у `mcp_server/server.py` — GitHub issue, пошук, URL, локальні нотатки |
| **Сторонній (опційно)** | `fetch` | Пакет `mcp-server-fetch` — загальний HTTP fetch (у `.env`: `FETCH_MCP_COMMAND`, `FETCH_MCP_ARGS`) |

Якщо `fetch` не підключено, для GitHub API використовується тул **`fetch_url`** з власного сервера.

## 2. Список тулів з описами (автоматично)

З кореня репозиторію (активуй `.venv`):

```bash
python main.py --list-mcp
```

Або окремий скрипт:

```bash
python scripts/mcp_inventory.py
```

Скрипт піднімає ті самі MCP, що й `main.py`, і для кожного сервера друкує **назву тула + docstring** з MCP.

## 3. Демо: референс → свій issue

Чернетка сценарію з конкретними тікетами (T001, T002, …) і що на кожному кроці показати: **[DEMO_TICKETS_UA.md](DEMO_TICKETS_UA.md)**.

**Референс (проєктний приклад):**

```bash
python main.py --url https://github.com/fastapi/fastapi/issues/1234 --no-hil
```

**Свій приклад** — заміни URL на публічний issue:

```bash
python main.py --url "https://github.com/ORG/REPO/issues/N" --no-hil
```

З підказкою типу задачі:

```bash
python main.py --url "https://github.com/ORG/REPO/issues/N" --hint duplicate --no-hil
```

Можливі значення `--hint`: `duplicate`, `code_area`, `stale`, `classify`.

Результат: блок **TRIAGE REPORT** у терміналі + файл `trajectories/<run_id>.json`.

## 4. Метрики по датасету проєкту

Датасет: **`evaluation/tasks.json`** (30 задач, різні категорії).

Повний прогін:

```bash
python main.py --eval --output trajectories
```

Після завершення:

- зведення в **`trajectories/summary.json`** (`score_pct`, `pass_rate`, `avg_latency_s`, `avg_tool_calls`, `by_category`, `tool_usage`, а також **`grounding`**, **`token_usage_est_total`**, **`error_tasks`**);
- друк у консолі блоку **EVALUATION COMPLETE** (ті самі метрики + grounding + токени + crash-и, якщо були).

Перегляд summary у зручному вигляді:

```bash
python scripts/show_eval_summary.py
# або
python scripts/show_eval_summary.py trajectories/summary.json
```

## 5. LangSmith (опційно)

Трасування вмикається тільки при `LANGCHAIN_TRACING_V2=true` та валідному `LANGCHAIN_API_KEY` — див. `.env.example`.
