# Розширені метрики

Модуль: **`agent/extended_metrics.py`**

## Один конкретний issue

### Після нового прогону

```powershell
cd C:\Users\Hello\Downloads\github-triage-agent
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil --extended-metrics
```

Опційно — критерії з eval-задачі T002 (той самий URL у датасеті):

```powershell
python main.py --url "https://github.com/fastapi/fastapi/issues/1663" --hint duplicate --no-hil --extended-metrics --task-id T002
```

У консолі з’явиться блок **«РОЗШИРЕНІ МЕТРИКИ»**. У файлі `trajectories/<uuid>.json` — ключ **`extended_metrics`**.

### Без повторного тріажу (існуючий JSON)

```powershell
python scripts/show_trajectory_metrics.py trajectories/T002.json --extended
```

## Eval (30 задач)

```powershell
python main.py --eval --prompt "Зроби мінімум 2 пошуки дублікатів і поясни висновок"
python main.py --eval --prompt-file prompts/duplicate.txt --tasks T002
```

Промпт з CLI застосовується до **кожної** задачі eval (додається в граф, як для `--url`).  
Якщо в `tasks.json` у задачі є `input.user_prompt` — він **перекриває** загальний `--prompt`.

Після `python main.py --eval` у кожному `trajectories/T00x.json` є **`extended_metrics`**.  
У `summary.json` → `triage_metrics.extended_metrics_summary` — середні по прогону.

## Групи метрик

| Група | Ключ JSON | Приклади полів |
|-------|-----------|----------------|
| Якість звіту | `report_quality` | `report_word_count`, `report_completeness_score`, `readability_score`, `actionability_score` |
| Класифікація | `classification` | `predicted_type`, `classification_confidence_score`, `type_correct_effective` (якщо `--task-id`) |
| Дублікати | `duplicate_search` | `search_query_count`, `unique_search_queries`, `similar_issues_found` |
| Tools | `tool_efficiency` | `tool_success_rate`, `tool_redundancy_rate`, `tools_by_name` |
| Стабільність | `stability` | `loop_count`, `budget_hit`, `error_recovery` |
| Галюцинації | `hallucination` | `grounding_passed`, `invented_fact_count`, `facts_grounded_rate` |
| Час/вартість | `cost_latency` | `duration_s`, `token_*_est`, `cost_per_task_usd_est` |
| Рубрика tasks.json | `rubric_autocheck` | `min_search_queries`, `must_flag_security`, … |

## Оцінка вартості

Змінні середовища (опційно):

- `METRICS_COST_INPUT_PER_1M` (default 1.25)
- `METRICS_COST_OUTPUT_PER_1M` (default 5.0)

## Скоринг eval (оновлено)

У **`evaluation/runner.py` → `score_result()`** додано автоматичні перевірки:

- `min_search_queries` — кількість викликів `search_similar_issues`
- `must_flag_security` — ключові слова security у звіті
- `must_compare_root_causes` — порівняння root cause у тексті

## Що поки евристика (не LLM)

- `readability_score`, `actionability_score`
- `duplicate_precision` — лише лічильники пошуку (без gold duplicate list)
- `consistency_score` — потрібні 2 прогони (не реалізовано)

Детальніше про архітектуру: [CODE_DOCUMENTATION_UA.md](./CODE_DOCUMENTATION_UA.md).
