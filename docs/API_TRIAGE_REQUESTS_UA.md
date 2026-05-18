# CLI і HTTP API: запити для тріажу по кожному типу

Один документ: **спочатку команди CLI**, **потім ті самі кейси як `POST /triage`** (JSON + коротко про `curl`).

**Умови:** з кореня репо, активний venv, заповнений `.env` (`GOOGLE_API_KEY` / `GEMINI_API_KEY`, `GITHUB_TOKEN`). Для стабільного прогону без зупинки на людині додано **`--no-hil`**.

**API:** база `http://localhost:8080`; у **`api.py`** приймаються лише URL, що починаються з **`https://github.com/`** (інакше **400**). Поля тіла: **`issue_url`** (обов’язково), **`task_hint`** (опційно: `classify` | `duplicate` | `code_area` | `stale` або `null` / не передавати).

---

# Частина 1 — CLI (`main.py`)

Шаблон:

```bash
python main.py --url "<ISSUE_URL>" --no-hil
python main.py --url "<ISSUE_URL>" --hint <classify|duplicate|code_area|stale> --no-hil
```

---

## 1. Класифікація (`classify`)

**З підказкою**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/10370" --hint classify --no-hil
```

**Без підказки**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/10370" --no-hil
```

Інші URL для демо того ж типу: `https://github.com/psf/requests/issues/6109`, `https://github.com/pallets/flask/issues/5153`.

---

## 2. Дублікати (`duplicate`)

**З підказкою**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --hint duplicate --no-hil
```

**Без підказки**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil
```

Інший приклад: `https://github.com/fastapi/fastapi/issues/3555`.

---

## 3. Зона коду (`code_area`)

**З підказкою**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/5920" --hint code_area --no-hil
```

**Без підказки**

```bash
python main.py --url "https://github.com/fastapi/fastapi/issues/5920" --no-hil
```

Інший приклад: `https://github.com/encode/httpx/issues/2714`.

---

## 4. Застарілість (`stale`)

**З підказкою**

```bash
python main.py --url "https://github.com/pallets/flask/issues/4179" --hint stale --no-hil
```

**Без підказки**

```bash
python main.py --url "https://github.com/pallets/flask/issues/4179" --no-hil
```

Інший приклад: `https://github.com/psf/requests/issues/6478`.

---

## 5. Крайові кейси (зазвичай лише CLI без «типової» підказки)

```bash
python main.py --url "https://github.com/pallets/flask/issues/4050" --no-hil
python main.py --url "https://github.com/fastapi/fastapi/issues/9999999" --no-hil
python main.py --url "https://github.com/fastapi/fastapi/pull/10000" --no-hil
python main.py --url "https://github.com/encode/httpx/issues/1234" --no-hil
```

Невалідний шлях на кшталт **`https://github.com/not-a-real-url`** зручніше саме в **CLI** (у HTTP його може відрізати валідація префікса).

---

# Частина 2 — HTTP API (`POST /triage`)

Після запуску сервера:

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Опитування результату: **`GET /triage/{job_id}`** (ід з відповіді POST). **`GET /health`**, **`GET /jobs`**.

---

## 1. Класифікація — API

**З підказкою**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/10370",
  "task_hint": "classify"
}
```

**Без підказки**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/10370"
}
```

---

## 2. Дублікати — API

**З підказкою**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/1663",
  "task_hint": "duplicate"
}
```

**Без підказки**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/1663"
}
```

---

## 3. Зона коду — API

**З підказкою**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/5920",
  "task_hint": "code_area"
}
```

**Без підказки**

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/5920"
}
```

---

## 4. Застарілість — API

**З підказкою**

```json
{
  "issue_url": "https://github.com/pallets/flask/issues/4179",
  "task_hint": "stale"
}
```

**Без підказки**

```json
{
  "issue_url": "https://github.com/pallets/flask/issues/4179"
}
```

---

## 5. Крайові кейси — API

Лише **`issue_url`** (без `task_hint`):

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/issues/9999999"
}
```

```json
{
  "issue_url": "https://github.com/fastapi/fastapi/pull/10000"
}
```

---

## Приклад `curl` (POST + опитування)

**bash**

```bash
curl -s -X POST http://localhost:8080/triage \
  -H "Content-Type: application/json" \
  -d '{"issue_url":"https://github.com/fastapi/fastapi/issues/1663","task_hint":"duplicate"}'
```

**PowerShell** (рядок тіла в екранованих лапках):

```powershell
curl.exe -s -X POST http://localhost:8080/triage `
  -H "Content-Type: application/json" `
  -d "{\"issue_url\": \"https://github.com/fastapi/fastapi/issues/1663\", \"task_hint\": \"duplicate\"}"
```

Далі (підстав свій `job_id` з JSON відповіді):

```bash
curl -s http://localhost:8080/triage/<job_id>
```

---

## Відповіді API

**`POST /triage`**

```json
{
  "job_id": "<uuid>",
  "status": "queued",
  "message": "Triage job started. Poll /triage/{job_id} for result."
}
```

**`GET /triage/{job_id}`** — поля: `status`, `task_type`, `report`, `grounding_passed`, `tool_calls`, `duration_s`, `error`.

---

## Де ще дивитися

- Сценарії демо та таблиці прикладів: **`docs/DEMO_TICKETS_UA.md`**
- MCP і метрики: **`docs/DEMO_AND_METRICS.md`**
