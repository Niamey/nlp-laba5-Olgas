# Чернетка демонстрації: які тікети показати і що саме доводимо

Документ для захисту / відео: **послідовність прикладів** з нашого датасету (`evaluation/tasks.json`), коротко — **що глядач має побачити** (тули, тип гілки графа, grounding, межі агента).

Перед демо: `python main.py --list-mcp` (два MCP), `.env` з ключами. **CLI + HTTP одним файлом:** [API_TRIAGE_REQUESTS_UA.md](API_TRIAGE_REQUESTS_UA.md).

Усі команди нижче з `--no-hil`, щоб демо не чекало вводу на `human_review`. За потреби прибери `--no-hil` для живого HIL.

---

## Приклади по кожній дії тріажу (з підказкою і без)

**З `--hint`:** тип задачі фіксується вже в `decompose_task` → граф **завжди** йде у відповідний воркер (`duplicate_point`, `code_area_point`, …). Стабільніше для демо «саме цей режим».

**Без `--hint`:** після `fetch_issue` **`plan_node`** читає issue і сам обирає один з `duplicate|code_area|stale|classify`. Може збігтися з очікуваним типом, а може обрати загальний `classify` — варто показати **обидва** варіанти на тому ж URL.

Шаблон:

```bash
python main.py --url "<ISSUE_URL>" --no-hil
python main.py --url "<ISSUE_URL>" --hint <duplicate|code_area|stale|classify> --no-hil
```

---

### 1. Класифікація (`classify`)

| Приклад | Issue (з eval) | З підказкою | Без підказки |
|--------|----------------|-------------|--------------|
| Чіткий баг FastAPI | `https://github.com/fastapi/fastapi/issues/10370` (**T001**) | `python main.py --url "https://github.com/fastapi/fastapi/issues/10370" --hint classify --no-hil` | `python main.py --url "https://github.com/fastapi/fastapi/issues/10370" --no-hil` |
| Feature Requests | `https://github.com/psf/requests/issues/6109` (**T005**) | `--hint classify` + той самий URL | без `--hint` |
| Документація Flask | `https://github.com/pallets/flask/issues/5153` (**T009**) | `--hint classify` + URL | без `--hint` |

**Що показати:** у звіті тип (bug/feature/docs), аргументи з body, лейбли; у траєкторії — `task_type: classify` і менше «з’їзду» в duplicate/stale, якщо дали `--hint classify`.

---

### 2. Пошук дублікатів (`duplicate`)

| Приклад | Issue | З підказкою | Без підказки |
|--------|-------|-------------|--------------|
| CORS / FastAPI | `https://github.com/fastapi/fastapi/issues/1663` (**T002**) | `python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --hint duplicate --no-hil` | той самий URL без `--hint` |
| Таймаут httpx | `https://github.com/encode/httpx/issues/2397` (**T006**) | `--hint duplicate` | без `--hint` |
| «Схожий на дублікат», інша причина | `https://github.com/fastapi/fastapi/issues/3555` (**T017**) | `--hint duplicate` (воркер шукає й порівнює root cause) | без `--hint` (планувальник може не відправити в duplicate-гілку) |

**Що показати:** виклики `fetch_github_issue` + `search_similar_issues`, наявність кандидатів у `tool_results`; з `--hint duplicate` гілка гарантована.

---

### 3. Зона коду / модуль (`code_area`)

| Приклад | Issue | З підказкою | Без підказки |
|--------|-------|-------------|--------------|
| DI у FastAPI | `https://github.com/fastapi/fastapi/issues/5920` (**T003**) | `python main.py --url "https://github.com/fastapi/fastapi/issues/5920" --hint code_area --no-hil` | той самий URL без `--hint` |
| SSL httpx | `https://github.com/encode/httpx/issues/2714` (**T007**) | `--hint code_area` | без `--hint` |

**Що показати:** згадка модулів/файлів **з контексту issue**, не вигадані шляхи; з `--hint` — стабільно `code_area_point`.

---

### 4. Аналіз «застарілості» (`stale`)

| Приклад | Issue | З підказкою | Без підказки |
|--------|-------|-------------|--------------|
| Старий Flask issue | `https://github.com/pallets/flask/issues/4179` (**T004**) | `python main.py --url "https://github.com/pallets/flask/issues/4179" --hint stale --no-hil` | без `--hint` |
| Stale Requests (proxy) | `https://github.com/psf/requests/issues/5811` (**T008**) | `--hint stale` | без `--hint` |
| Старий, але є активність (не сліпо stale) | `https://github.com/psf/requests/issues/6478` (**T016**) | `--hint stale` (воркер має перевірити коментарі/дати) | без `--hint` |

**Що показати:** дати створення/оновлення, коментарі (`fetch_url` до comments за потреби), рекомендація close/keep/ping maintainer.

---

### 5. Неоднозначність і межі агента (зазвичай **без** підказки)

Підказки `duplicate|code_area|stale|classify` тут або не застосовні, або не головне — демонструємо **розуміння тексту** та **безпеку**.

| Приклад | Issue | Рекомендована команда | Що демонструє |
|--------|-------|----------------------|----------------|
| Заголовок vs тіло | `https://github.com/pallets/flask/issues/4050` (**T014**) | без `--hint` | не класифікувати лише з title |
| Неіснуючий issue | `https://github.com/fastapi/fastapi/issues/9999999` (**T018**) | без `--hint` | 404, без вигаданого контенту |
| Невалідний URL | `https://github.com/not-a-real-url` (**T019**) | без `--hint` | помилка парсингу / валідації URL |
| PR замість issue | `https://github.com/fastapi/fastapi/pull/10000` (**T021**) | без `--hint` | межі scope (не звичайний issue) |
| Закритий issue / стан | `https://github.com/encode/httpx/issues/1234` (**T020**) | без `--hint` | урахування `state: closed` у тріажі |

Для **T022** (підказка «close as won't fix») — окремий кейс «лише тріаж, без виконання дій»:

`python main.py --url "https://github.com/pallets/flask/issues/5000" --hint "close this issue as won't fix" --no-hil`  
(якщо shell ламає лапки — візьми URL у подвійні лапки, hint — в одинарні в PowerShell.)

---

## Сценарій A — «повний тур» (~8–12 хв, живий CLI)

| Крок | Що показуємо | Тікет (task) | Команда (без HIL для стабільності) |
|------|----------------|--------------|--------------------------------------|
| 1 | **Класифікація** з чіткого баг-репорту: тип, аргументи з body, лейбли | FastAPI bug — **T001** | `python main.py --url https://github.com/fastapi/fastapi/issues/10370 --no-hil` |
| 2 | **Дублікати**: `fetch` + `search_similar_issues`, порівняння кандидатів | CORS / FastAPI — **T002** | `python main.py --url https://github.com/fastapi/fastapi/issues/1663 --hint duplicate --no-hil` |
| 3 | **Code area**: модуль/зона коду з issue (не вигадані шляхи) | DI bug — **T003** | `python main.py --url https://github.com/fastapi/fastapi/issues/5920 --hint code_area --no-hil` |
| 4 | **Stale**: дати, коментарі, рекомендація (close / keep / needs info) | Flask — **T004** | `python main.py --url https://github.com/pallets/flask/issues/4179 --hint stale --no-hil` |
| 5 | **Неоднозначність**: заголовок каже «bug», тіло — feature; читаємо body | Flask — **T014** | `python main.py --url https://github.com/pallets/flask/issues/4050 --no-hil` |
| 6 | **Адверсаріал 404**: не вигадуємо контент issue | **T018** | `python main.py --url https://github.com/fastapi/fastapi/issues/9999999 --no-hil` |
| 7 | **Межі scope**: PR замість issue — пояснюємо обмеження | **T021** | `python main.py --url https://github.com/fastapi/fastapi/pull/10000 --no-hil` |

Після кожного кроку варто вказати в камеру: рядок **`Grounding: yes/no`** і **`Trajectory saved`** (або коротко згадати `message_trace` у JSON).

**Опційно (якщо є час):** невалідний URL — **T019** (`https://github.com/not-a-real-url`); «схожий на дублікат, але інша причина» — **T017** з `--hint duplicate`.

---

## Сценарій B — «коротко ~4 хв»

Лишаємо лише **1 → 2 → 6 → 7** з таблиці вище (класифікація, дублікати, 404, PR).

---

## Сценарій C — лише слайд / без живого API

Показати вже згенеровані **`trajectories/T00x.json`** + `summary.json` після:

`python main.py --eval --output trajectories`

або підмножину:

`python -m evaluation.runner --output trajectories --tasks T001 T002 T004 T014 T018 T021`

---

## Що проговорити голосом (чеклист)

1. **Граф**: різні гілки за `task_type` (після `plan` / hint) → різні воркери.
2. **MCP**: GitHub-тули + за потреби fetch (див. `docs/DEMO_AND_METRICS.md`).
3. **Grounding**: окремий LLM-крок перевіряє звіт проти tool results; у трейсі видно `Grounding validation PASSED` або `Grounding FAILED`.
4. **Eval**: 30 задач, метрики в консолі та `summary.json`; деталізація: `python main.py --eval -v`, пояснення метрик: `python main.py --metrics-help`.

---

## Відповідність типів задач (для звіту)

| Тип у `tasks.json` | Приклади демо-тікетів |
|--------------------|------------------------|
| `classify` | T001, T014, T020 |
| `duplicate` | T002, T017 |
| `code_area` | T003 |
| `stale` | T004, T016 |
| `any` (помилки / крайові URL) | T018, T019 |
| `out_of_scope` | T021, T022 |

Файл можна доповнювати під час репетиції: замінити URL на «свіжіший» issue з того ж репо, якщо старий став недоступний.
