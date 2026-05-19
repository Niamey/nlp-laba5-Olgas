# Розширені метрики тріажу — опис змін

Документ описує **що було додано**, **які метрики** збираються, **як вони рахуються**, **де в коді** це реалізовано та **як запускати** на одному issue і в eval.

Пов’язані файли: [EXTENDED_METRICS_UA.md](./EXTENDED_METRICS_UA.md) (короткий гайд), [CODE_DOCUMENTATION_UA.md](./CODE_DOCUMENTATION_UA.md) (архітектура проєкту).

---

## 1. Мета змін

До цих змін після одного прогону (`python main.py --url …`) були лише базові поля trajectory: `tool_calls`, `grounding_passed`, `duration_s`, `token_usage_est`.  

Після eval (`--eval`) у `summary.json` були `score_pct`, `pass_rate`, `triage_metrics`, але **не було детальної аналітики якості звіту, пошуку дублікатів і ефективності tools на рівні одного прогону**.

**Ціль:** додати блок **`extended_metrics`** — структуровані метрики одного тріажу + зручний вивід у консоль + агрегат по eval.

---

## 2. Що було зроблено (перелік змін)

| № | Зміна | Файл |
|---|--------|------|
| 1 | Новий модуль розрахунку та друку метрик | `agent/extended_metrics.py` (**новий**) |
| 2 | CLI `--extended-metrics`, `--task-id`; запис у trajectory JSON | `main.py` |
| 3 | `extended_metrics` у кожній задачі eval; `extended_metrics_summary` у summary | `evaluation/runner.py` |
| 4 | Вивід метрик з готового JSON без API | `scripts/show_trajectory_metrics.py` (`--extended`) |
| 5 | Середні розширених метрик у pretty-summary | `scripts/show_eval_summary.py` |
| 6 | Авто-скоринг: `min_search_queries`, `must_flag_security`, `must_compare_root_causes` | `evaluation/runner.py` → `score_result()` |
| 7 | Промпт користувача на eval: `--prompt`, `--prompt-file`, `--tasks` | `main.py`, `evaluation/runner.py` |
| 8 | Завантаження промпта з CLI/файлу | `agent/user_instructions.py` |
| 9 | Юніт-тести метрик | `tests/test_extended_metrics.py` (**новий**) |
| 10 | Виправлення синтаксису в decompose | `agent/nodes/decompose.py` (`F"""` → `"""`) |

---

## 3. Додані та змінені файли

### 3.1. Нові файли

```
agent/extended_metrics.py
tests/test_extended_metrics.py
docs/EXTENDED_METRICS_UA.md
docs/ROZSHYRENI_METRYKY_ZMINY_UA.md   ← цей документ
```

### 3.2. Змінені файли

```
main.py
evaluation/runner.py
scripts/show_trajectory_metrics.py
scripts/show_eval_summary.py
agent/user_instructions.py
agent/nodes/decompose.py
docs/DEMO_AND_METRICS.md
docs/CODE_DOCUMENTATION_UA.md
README.md
```

---

## 4. Структура `extended_metrics` (JSON)

Після прогону в `trajectories/<uuid>.json` або `trajectories/T00x.json`:

```json
{
  "extended_metrics": {
    "report_quality": { ... },
    "classification": { ... },
    "duplicate_search": { ... },
    "tool_efficiency": { ... },
    "stability": { ... },
    "hallucination": { ... },
    "cost_latency": { ... },
    "rubric_autocheck": { ... }
  }
}
```

У `trajectories/summary.json` після eval додатково:

```json
{
  "triage_metrics": {
    "extended_metrics_summary": {
      "tasks_with_extended_metrics": 30,
      "avg_report_completeness": 0.95,
      "hallucination_rate": 0.1,
      ...
    }
  }
}
```

---

## 5. Метрики: що означають і як рахуються

Джерело даних для розрахунку:

- `report` — текст `triage_report`;
- `state` — фінальний стан LangGraph після прогону;
- `trajectory` — експорт з `main.run_triage()` (`tool_results`, `grounding_passed`, `fact_check`, `token_usage_est`, …);
- `task` — опційно запис з `evaluation/tasks.json` (для gold type і `rubric_autocheck`).

Функція-збірка: **`compute_extended_metrics()`** у `agent/extended_metrics.py`.

---

### 5.1. `report_quality` — якість звіту

**Функція:** `compute_report_quality(report)`

| Поле | Як рахується |
|------|----------------|
| `report_word_count` | `len(report.split())` |
| `report_char_count` | `len(report)` |
| `sections_found` | Regex по markdown-заголовках: Summary, Analysis, Recommendation, Labels |
| `report_completeness_score` | Частка з 3 обов’язкових секцій (summary, analysis, recommendation): 0.0–1.0 |
| `avg_sentence_words` | Середня довжина речення (розбиття по `.!?`) |
| `readability_score` | Евристика: 1.0 − \|avg_words − 16\| / 24, обрізано до [0, 1] |
| `actionability_score` | min(1, кількість збігів ключових слів / 3); ключі: recommend, suggest, close, duplicate of, … |

**Не використовується:** LLM-оцінка читабельності (Flesch-Kincaid можна додати окремо).

---

### 5.2. `classification` — класифікація

**Функція:** `compute_classification_metrics(state, report, task)`

| Поле | Як рахується |
|------|----------------|
| `predicted_type` | `state.triage_findings.classification.type` або `duplicate`, якщо `is_duplicate` |
| `classification_confidence` | `high` / `medium` / `low` з findings |
| `classification_confidence_score` | Мапа: high→1.0, medium→0.66, low→0.33 |
| `gold_types`, `type_match_*`, `type_correct_effective` | Лише якщо передано `task` з `success_criteria.type_correct` |

**Примітка:** для гілки duplicate без блоку `classification` у state поле `predicted_type` може бути `null` — це очікувано.

---

### 5.3. `duplicate_search` — пошук дублікатів

**Функція:** `compute_duplicate_search_metrics(state, trajectory)`

| Поле | Як рахується |
|------|----------------|
| `search_query_count` | Кількість елементів у `tool_results` з `"tool": "search_similar_issues"` |
| `unique_search_queries` | Унікальні значення `args.query` |
| `search_queries` | Список запитів (до 10) |
| `similar_issues_found` | `len(state.similar_issues)` |
| `duplicate_of_predicted` | `triage_findings.duplicate.duplicate_of` |
| `search_tool_called` | `search_query_count > 0` |

**Не рахується автоматично:** `duplicate_precision` (потрібен gold-список дублікатів).

---

### 5.4. `tool_efficiency` — інструменти MCP

**Функція:** `compute_tool_efficiency(state, trajectory)`

| Поле | Як рахується |
|------|----------------|
| `tool_calls_total` | `len(tool_results)` |
| `tool_success_rate` | (n − помилки) / n; помилка = `result.error` або "failed" у тексті |
| `tool_redundancy_rate` | 1 − (унікальні сигнатури tool+args) / n |
| `tools_by_name` | Підрахунок по полю `tool` |
| `tokens_per_tool_call` | `(input+output tokens) / tool_calls_total` (у `compute_extended_metrics`) |

---

### 5.5. `stability` — стабільність прогону

**Функція:** `compute_stability_metrics(state, trajectory, report)`

| Поле | Як рахується |
|------|----------------|
| `loop_count` | `state.loop_count` або `trajectory.loop_count` |
| `error_count` | `state.error_count` |
| `retry_like_loop` | `loop_count > 0` |
| `budget_hit` | У тексті звіту є "budget limit" / "budget exceeded" |
| `error_recovery` | Є звіт і `error_count > 0` |

**Не реалізовано:** `consistency_score` (потрібні два прогони на той самий URL).

---

### 5.6. `hallucination` — галюцинації та grounding

**Функція:** `compute_hallucination_metrics(trajectory, state)`

Модуль **не рахує** grounding заново — **читає** результат валідації.

| Поле | Джерело |
|------|---------|
| `grounding_passed` | `trajectory.grounding_passed` / `state.grounding_passed` |
| `fact_check_*` | `trajectory.fact_check` (regex у `agent/fact_extractor.py`) |
| `hallucination_detected` | `len(invented) > 0` або `grounding_passed == false` |
| `invented_fact_count` | `len(fact_check.invented)` |

**Де рахується grounding (окремо від extended_metrics):**

- `agent/nodes/completion.py` → `grounding_validator()`:
  1. LLM порівнює звіт з ground-truth blob;
  2. Regex `check_facts_against_blob()` — вигадані #issue, @user, paths, URL;
  3. Опційно другий LLM (`TRIAGE_DOUBLE_GROUNDING=1`).

---

### 5.7. `cost_latency` — час і вартість

**Функція:** `compute_cost_latency(trajectory, state)`

| Поле | Як рахується |
|------|----------------|
| `duration_s` | `trajectory.duration_s` |
| `token_*_est` | `trajectory.token_usage_est` з `agent/trajectory_export.aggregate_token_usage()` |
| `cost_per_task_usd_est` | `(input×PIN + output×POUT) / 1e6`; env: `METRICS_COST_INPUT_PER_1M` (default 1.25), `METRICS_COST_OUTPUT_PER_1M` (default 5.0) |

---

### 5.8. `rubric_autocheck` — підказки з tasks.json

**Функція:** `compute_rubric_hints(report, state, trajectory, task)`

Працює лише якщо передано `task` (наприклад `--task-id T002`).

| Перевірка | Логіка |
|-----------|--------|
| `min_search_queries` | `len(search_similar_issues calls) >= required` |
| `must_flag_security` | Є підрядок з security/cve/vulnerability/… у звіті |
| `must_compare_root_causes` | Є root cause / compare / versus / … у звіті |

Це **попередній перегляд**; офіційний бал eval — у `score_result()`.

---

## 6. Де в коді (карта викликів)

```
┌─────────────────────────────────────────────────────────────┐
│  agent/extended_metrics.py                                  │
│    compute_report_quality, compute_duplicate_search_*, …    │
│    compute_extended_metrics()  ← головна точка               │
│    print_extended_metrics()    ← консольний вивід            │
│    aggregate_extended_metrics_summary()  ← eval aggregate    │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   main.py            evaluation/runner.py   show_trajectory_metrics.py
   --extended-metrics  після кожної задачі   --extended
   → trajectory JSON   → T00x.json + summary
```

### 6.1. `agent/extended_metrics.py` (новий)

| Рядки (орієнтовно) | Що |
|--------------------|-----|
| 17–36 | Regex секцій звіту |
| 95–120 | `compute_report_quality` |
| 123–163 | `compute_classification_metrics` |
| 166–187 | `compute_duplicate_search_metrics` |
| 190–216 | `compute_tool_efficiency` |
| 219–229 | `compute_stability_metrics` |
| 232–251 | `compute_hallucination_metrics` |
| 254–271 | `compute_cost_latency` |
| 274–306 | `compute_rubric_hints` |
| 309–347 | **`compute_extended_metrics`** |
| 350–403 | `aggregate_extended_metrics_summary` |
| 406–467 | **`print_extended_metrics`** |

### 6.2. `main.py`

| Що | Де |
|----|-----|
| `--extended-metrics` | argparse |
| `--task-id T00x` | підстановка критеріїв з `tasks.json` |
| `compute_extended_metrics` + збереження | після `run_triage`, перед/після запису JSON |
| `print_extended_metrics` | якщо `--extended-metrics` |

### 6.3. `evaluation/runner.py`

| Що | Де |
|----|-----|
| `extended = compute_extended_metrics(...)` | цикл по задачах eval |
| `entry["extended_metrics"]` | результат задачі |
| `triage_metrics["extended_metrics_summary"]` | після агрегації |
| `min_search_queries`, `must_flag_security`, `must_compare_root_causes` | у `score_result()` |
| `user_prompt` на eval | параметр `run_evaluation(user_prompt=…)` |

### 6.4. Залежності (не в extended_metrics, але потрібні для полів)

| Поле в trajectory | Файл |
|-------------------|------|
| `grounding_passed`, `fact_check` | `agent/nodes/completion.py` |
| `token_usage_est` | `agent/trajectory_export.py` |
| `tool_results`, `similar_issues` | вузли `agent/nodes/workers.py`, `fetch.py` |

---

## 7. Відмінність: один issue vs eval

| | Один `--url` | `--eval` |
|---|--------------|----------|
| Файл результату | `trajectories/<uuid>.json` | `trajectories/T001.json` … + `summary.json` |
| Ключ метрик | `extended_metrics` | `extended_metrics` у кожному T00x + `extended_metrics_summary` |
| Score (бали) | Немає (окрім `rubric_autocheck` з `--task-id`) | `score`, `passed`, `score_pct` |
| CLI | `--extended-metrics` | той самий модуль після кожної задачі |

---

## 8. Як запустити

### 8.1. Один issue — метрики в консоль + JSON

```powershell
cd C:\Users\Hello\Downloads\github-triage-agent
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" ^
  --hint duplicate --no-hil --extended-metrics --task-id T002
```

З користувацьким промптом:

```powershell
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil ^
  --extended-metrics --prompt "Зроби мінімум 2 пошуки дублікатів і поясни висновок"
```

Зберегти все в один файл:

```powershell
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil ^
  --extended-metrics --metrics-out run_metrics.json
```

### 8.2. Перегляд без повторного API

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/show_trajectory_metrics.py trajectories/T002.json --extended
```

### 8.3. Eval з розширеними метриками

```powershell
python main.py --eval --no-hil
python main.py --eval --no-hil --prompt "Зроби мінімум 2 пошуки дублікатів"
python main.py --eval --tasks T002 T011 --prompt-file prompts/duplicate.txt
```

Після eval:

```powershell
python scripts/show_eval_summary.py
```

---

## 9. Приклад виводу (один прогон, issue #1663)

```
══════════════════════════════════════════════════════════════
  РОЗШИРЕНІ МЕТРИКИ (один прогон)
  Issue: https://github.com/fastapi/fastapi/issues/1663
══════════════════════════════════════════════════════════════

  ■ Якість звіту
    Слів: 136  |  Символів: 1030
    Повнота секцій: 1.0
    Readability: 0.958  |  Actionability: 1.0

  ■ Пошук дублікатів
    search_similar_issues викликів: 3  |  унікальних запитів: 3

  ■ Інструменти
    Всього викликів: 4  |  success rate: 1.0

  ■ Grounding
    grounding_passed: True  |  invented facts: 0

  ■ Час і вартість
    duration_s: 27.82  |  tokens: 6892/4362/11254
```

---

## 10. Обмеження та майбутні покращення

| Метрика з плану | Статус |
|-----------------|--------|
| report_length, completeness, readability, actionability | Реалізовано (евристики) |
| classification_confidence | Реалізовано |
| precision/recall/F1 по типах | Не реалізовано (потрібен batch по eval) |
| duplicate_precision, duplicate_rank | Не реалізовано (потрібен gold) |
| LLM actionability_score | Не реалізовано |
| consistency_score (2 прогони) | Не реалізовано |
| false_negative_grounding (LLM) | Не реалізовано |
| p99_latency у extended_metrics | Є в `triage_metrics.latency_s` після eval, не в одному прогоні |

Критерії з `tasks.json`, які **є в JSON**, але **раніше не входили в `score_result()`** — частково закрито: `min_search_queries`, `must_flag_security`, `must_compare_root_causes`. Інші ключі (`must_acknowledge_uncertainty`, `must_read_body`, …) досі лише в rubric-тексті, без автоматичного штрафу в коді.

---

## 11. Тести

```powershell
pytest tests/test_extended_metrics.py -v
```

Перевіряє: повноту секцій звіту, мінімальний `compute_extended_metrics`, `min_search_queries` у rubric_autocheck.

---

*Документ описує стан коду після додавання модуля `agent/extended_metrics.py` та інтеграції в CLI/eval.*
