# Архітектура по флоу: загальна, LangChain/LangGraph, MCP

У теці **`docs/architecture/`** лежать діаграми **Mermaid** (`.mmd`) і згенеровані **`*.svg` / `*.png`** (якщо їх зібрано через `@mermaid-js/mermaid-cli`). Нижче — **текстовий опис кожного флоу** для захисту.

| Файл діаграми | Флоу |
|---------------|------|
| `00-general-flow` | Загальна архітектура end-to-end |
| `05-langchain-flow` | LangChain + LangGraph |
| `06-mcp-flow` | MCP: конфіг, процеси, тулі в агенті |

Додатково: `01-overview`, `02-langgraph`, `03-mcp-github-triage`, `04-mcp-fetch` — деталізація (див. `README.md` у тій же теці).

---

## Флоу 0 — загальна архітектура (`00-general-flow`)

**Що показує:** один **скрізний сценарій** від людини до результату, без занурення в реалізацію вузлів.

**Кроки:**

1. **Вхід** — користувач або клієнт (термінал, браузер, Swagger, скрипт).
2. **Точка входу** — або **`main.py`** (CLI), або **`api.py`** (HTTP). У випадку API спочатку створюється **задача** (`job_id`), а важка робота йде у фоні.
3. **`run_triage`** (`main.py`) — єдина «склейка»: завантажити MCP-тулі, підготувати конфіг графа, викликати **`graph.ainvoke`**.
4. **Паралельні підсистеми (логічно):**
   - **LangGraph** — кроки міркування, стан, розгалуження, звіт, grounding.
   - **MCP** — окремі процеси з інструментами; граф звертається до них під час кроків.
   Зв’язок **LangGraph ↔ MCP**: вузли графа викликають тулі; відповіді потрапляють у **`tool_results`** у стані.
5. **Вихід** — текст **`triage_report`**, **`trajectory`** (метадані, grounding, лічильники). Для API — оновлення запису задачі та **GET** по `job_id`.

**Навіщо окремий флоу:** на захисті можна за 1 хвилину пояснити **роль кожного шару**, не відкриваючи десятки файлів.

---

## Флоу LangChain / LangGraph (`05-langchain-flow`)

**Що показує:** як **оркеструється міркування** і де з’являється **LLM**, без деталей протоколу MCP.

### LangChain (у проєкті)

- **`ChatGoogleGenerativeAI`** у **`agent/llm.py`** (`get_llm`, `get_fast_llm`) — виклики моделі з промптами.
- **Повідомлення** LangChain (`SystemMessage`, `HumanMessage`, `AIMessage`, …) формуються у вузлах (**`planner`**, **`completion`**, воркери тощо) і накопичуються в **`messages`** у стані (**`IssueTriageState`**).

### LangGraph

- **`initial_state`** — стартовий словник стану (**`agent/state.py`**).
- **Лінійна частина початку:** `decompose_task` → `fetch_issue` → `plan_node` — підготовка даних і плану.
- **Роутер** (`agent/nodes/router.py`) — вибір воркера за **`task_type`** з обмеженням бюджету.
- **Воркери** — збір **`triage_findings`** під конкретний тип triage.
- **`draft_answer`** — синтез Markdown-звіту через LLM.
- **`grounding_validator`** — друга LLM-перевірка узгодженості з даними; оновлює **`grounding_passed`**.
- **`completion_check`** — повнота звіту, цикл **`needs_more_info`** назад до роутера або **`human_review`** (HIL), або вихід у **`END`**.

**Навіщо окремий флоу:** відокремити **«як думає граф»** від **«звідки беруться факти»** (MCP).

---

## Флоу MCP (`06-mcp-flow`)

**Що показує:** як **піднімаються та використовуються** інструменти, а не внутрішня реалізація кожного HTTP-запиту.

### Конфігурація

- **`build_mcp_connections`** / **`mcp_config_for_main`** (`agent/mcp_config.py`) — словник: ім’я сервера → `command` + `args` + `transport` (stdio) + `env` для `github_triage`.

### Два процеси

1. **`github_triage`** — запуск **`mcp_server/server.py`** (FastMCP). Тулі: issue, GitHub API URL, пошук схожих, нотатки.
2. **`fetch`** (опційно) — з `.env` (`FETCH_MCP_*`); пакет **`mcp-server-fetch`**, тул **`fetch`**.

### У процесі `run_triage`

- **`MultiServerMCPClient`** завантажує тулі з обох процесів.
- **`set_mcp_tools`** (`agent/mcp_tools_context.py`) зберігає словник **ім’я → callable** на час **`ainvoke`**, щоб вузли графа викликали **ті самі** інструменти.

### У вузлах

- Воркери та **`fetch_issue`** звертаються до тулів; результати додаються в **`tool_results`** у **`IssueTriageState`**.

**Навіщо окремий флоу:** на захисті чітко видно **multi-server MCP**, **stdio** і **місце**, де тулі стають доступними Python-коду графа.

---

## Як згенерувати картинки з нових `.mmd`

```bash
cd docs/architecture
npx --yes @mermaid-js/mermaid-cli -i 00-general-flow.mmd -o 00-general-flow.png
npx --yes @mermaid-js/mermaid-cli -i 05-langchain-flow.mmd -o 05-langchain-flow.png
npx --yes @mermaid-js/mermaid-cli -i 06-mcp-flow.mmd -o 06-mcp-flow.png
```

(Аналогічно `-o … .svg` для вектора.)
