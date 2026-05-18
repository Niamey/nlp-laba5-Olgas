# GitHub Issue Triage Agent

**NLP Assignment #5 — Agentic Systems | Track C**

LangGraph + MCP агент для автоматичного тріажу GitHub issues.

---

## Архітектура

```
START → decompose_task → fetch_issue → plan_node → route_use_selector
                                                          │
                    ┌──────────┬──────────┬──────────┬───┘
                    ▼          ▼          ▼          ▼
             duplicate_pt  code_area  stale_pt  classify_pt
                    └──────────┴──────────┴──────────┘
                                    │
                              draft_answer
                                    │
                          grounding_validator
                                    │
                           completion_check
                            ┌───────┼───────┐
                            ▼       ▼       ▼
                      human_review  loop  END
```

**Ключові вимоги:**
- `interrupt_before=["human_review"]` → HIL interrupt для demo
- `MemorySaver` checkpointer → persistent state
- Conditional edges: `route_use_selector`, `completion_check`
- 2 MCP сервери: власний (`mcp_server/server.py`) + `mcp-server-fetch`

---

## Швидкий старт

### 1. Встановлення

```bash
git clone <repo>
cd github-triage-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Конфігурація

```bash
cp .env.example .env
# Заповни GOOGLE_API_KEY або GEMINI_API_KEY, і GITHUB_TOKEN
```

### 3. Запуск одного issue

```bash
python main.py --url https://github.com/fastapi/fastapi/issues/1234
```

З підказкою:
```bash
python main.py --url https://github.com/pallets/flask/issues/5000 --hint duplicate
```

Без HIL (для скриптів):
```bash
python main.py --url URL --no-hil
```

### 4. Запуск evaluation suite

```bash
python main.py --eval --output trajectories/
```

### Демо, список MCP і метрики (захист)

- Повний сценарій: **[docs/DEMO_AND_METRICS.md](docs/DEMO_AND_METRICS.md)**
- Показати всі MCP-сервери й тулі з описами: `python main.py --list-mcp`
- Останній `summary.json` після eval: `python scripts/show_eval_summary.py`
- Перевірка набору задач перед eval: `python scripts/check_assignment_readiness.py`
- Шаблон письмового звіту (UA): **[docs/ZVIT_TRACK_C_UA.md](docs/ZVIT_TRACK_C_UA.md)**

Після одиночного або eval-прогону JSON-траєкторія містить `message_trace` (журнал повідомлень із прев’ю та metadata) та `token_usage_est` для аналізу токенів без LangSmith.

### 5. Аблації

```bash
# Всі 3 аблації
python -m evaluation.ablations --ablation all

# Окремо
python -m evaluation.ablations --ablation model
python -m evaluation.ablations --ablation prompt
python -m evaluation.ablations --ablation graph
```

### 6. HTTP API

Приклади **CLI**, потім **HTTP JSON** для кожного типу тріажу: **[docs/API_TRIAGE_REQUESTS_UA.md](docs/API_TRIAGE_REQUESTS_UA.md)**.

```bash
uvicorn api:app --port 8080

# Запустити тріаж
curl -X POST http://localhost:8080/triage \
  -H "Content-Type: application/json" \
  -d '{"issue_url": "https://github.com/fastapi/fastapi/issues/1234"}'

# Отримати результат
curl http://localhost:8080/triage/{job_id}
```

### 7. Demo (для захисту)

```bash
# Термінал 1 — запуск з HIL
python main.py --url https://github.com/fastapi/fastapi/issues/1234

# Коли агент зупиниться на human_review:
# → Переглянь звіт
# → Введи 'approve' або фідбек для корекції
```

---

## Деплой

### Docker

```bash
docker build -t github-triage-agent .
docker run -e GOOGLE_API_KEY=... -e GITHUB_TOKEN=... github-triage-agent \
  python main.py --url https://github.com/fastapi/fastapi/issues/1234 --no-hil
```

### Docker Compose

```bash
ISSUE_URL=https://github.com/fastapi/fastapi/issues/1234 docker compose up
```

Evaluation:
```bash
docker compose --profile eval up eval
```

### Railway

```bash
npm install -g @railway/cli
railway login
railway init
# Додай змінні: GOOGLE_API_KEY або GEMINI_API_KEY, GITHUB_TOKEN
railway up
```

### Render

1. New Web Service → Connect repo
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Add env vars у dashboard

---

## MCP Сервери

### Власний (`mcp_server/server.py`)

| Tool | Опис |
|------|------|
| `fetch_github_issue` | Завантажує issue з GitHub API (кеш 6h) |
| `search_similar_issues` | Пошук схожих issues в репо (кеш) |
| `fetch_url` | Довільний GitHub API URL |
| `save_triage_note` | Зберігає нотатку (SQLite) |
| `get_triage_notes` | Читає нотатки для issue |

Запуск окремо:
```bash
GITHUB_TOKEN=ghp_... python mcp_server/server.py
```

### Третій сторонній (`mcp-server-fetch`)

```bash
pip install mcp-server-fetch
# або: uvx mcp-server-fetch
```

У `.env` (опціонально): `FETCH_MCP_COMMAND=uvx` та `FETCH_MCP_ARGS=mcp-server-fetch`. Якщо не задаєш — використовується лише інструмент `fetch_url` з власного MCP.

---

## Evaluation Set

30 задач у `evaluation/tasks.json`, покривають:

| Категорія | Кількість | Опис |
|-----------|-----------|------|
| `happy_path` | 12 | Стандартні задачі тріажу |
| `ambiguous`  | 5  | Неоднозначні, потребують нюансів |
| `adversarial` | 3 | Хибні передумови, неіснуючі issues |
| `should_refuse` | 3 | Агент має відмовити або скорегувати scope |

Рубрика: 0-3 бали за задачу (0=fail, 1=partial, 2=good, 3=excellent)

---

## Репозиторії для Evaluation

1. `fastapi/fastapi` — FastAPI framework
2. `pallets/flask` — Flask framework
3. `psf/requests` — Requests library
4. `encode/httpx` — HTTPX async client
5. `tiangolo/sqlmodel` — SQLModel ORM

---

## Структура проєкту

```
github-triage-agent/
├── agent/
│   ├── state.py          # TypedDict state schema
│   ├── graph.py          # LangGraph StateGraph
│   ├── llm.py            # LLM configuration
│   └── nodes/
│       ├── decompose.py  # URL parsing
│       ├── fetch.py      # GitHub API fetch
│       ├── planner.py    # Plan + task_type detection
│       ├── router.py     # Routing functions
│       ├── workers.py    # 4 specialized workers
│       └── completion.py # Draft + validate + HIL
├── mcp_server/
│   ├── server.py         # Custom MCP server (separate process)
│   ├── Dockerfile
│   └── requirements.txt
├── evaluation/
│   ├── tasks.json        # 30 evaluation tasks
│   ├── runner.py         # Batch evaluation runner
│   └── ablations.py      # 3 ablation studies
├── main.py               # CLI entry point
├── api.py                # FastAPI HTTP server
├── Dockerfile
├── docker-compose.yml
├── railway.toml
└── requirements.txt
```

---

## Ліміти та бюджет

| Параметр | Значення |
|----------|---------|
| `MAX_TOOL_CALLS` | 25 per run |
| `MAX_LOOPS` | 3 |
| Cache TTL | 6 годин |
| Rate limit delay | 0.5s між GitHub запитами |

---

## LangSmith трасування

```bash
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=github-triage-agent
```

Кожен run зберігає trajectory у `trajectories/{run_id}.json`.
