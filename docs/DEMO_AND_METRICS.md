# Демо та метрики (захист / доповідь)

**Детальна документація по коду** (граф, скоринг, усі метрики, посилання на файли/рядки):  
→ [CODE_DOCUMENTATION_UA.md](./CODE_DOCUMENTATION_UA.md)

**Опис змін: розширені метрики** (що додано, як рахуються, де в коді):  
→ [ROZSHYRENI_METRYKY_ZMINY_UA.md](./ROZSHYRENI_METRYKY_ZMINY_UA.md)

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

З **користувацьким промптом** на весь eval (або лише кілька задач):

```bash
python main.py --eval --no-hil --prompt "Знайди схожі issues, чи це дублікат"
python main.py --eval --prompt-file prompts/duplicate.txt --tasks T002 T011
```

Один issue з промптом (як раніше):

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil --prompt "…"
```

Після завершення:

- зведення в **`trajectories/summary.json`** (`score_pct`, `pass_rate`, `avg_latency_s`, `avg_tool_calls`, `by_category`, `tool_usage`);
- друк у консолі блоку **EVALUATION COMPLETE**.

Перегляд summary у зручному вигляді:

```bash
python scripts/show_eval_summary.py
# або
python scripts/show_eval_summary.py trajectories/summary.json
```

## 5. LangSmith (опційно)

Трасування вмикається тільки при `LANGCHAIN_TRACING_V2=true` та валідному `LANGCHAIN_API_KEY` — див. `.env.example`.
