# Приклади для ручного тестування

Реальні публічні issues (як у eval). Запускай з кореня репозиторію, з активованим venv.

## Підказка (`--hint`) + промпт (`hint-prompt`)

```bash
python main.py --launch-mode hint-prompt --hint duplicate --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil --prompt-file prompts/testing/hint_duplicate_uk.txt

python main.py --launch-mode hint-prompt --hint classify --url "https://github.com/fastapi/fastapi/issues/10370" --no-hil --prompt-file prompts/classify.txt

python main.py --launch-mode hint-prompt --hint code_area --url "https://github.com/fastapi/fastapi/issues/5920" --no-hil --prompt-file prompts/code_area.txt

python main.py --launch-mode hint-prompt --hint stale --url "https://github.com/pallets/flask/issues/4179" --no-hil --prompt-file prompts/stale.txt
```

Інлайн-промпт (без файлу):

```bash
python main.py --launch-mode hint-prompt --hint duplicate --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil --prompt "Зроби два пошуки схожих issues, у звіті вкажи номери та вердикт: дублікат / related / унікальний."
```

## Лише промпт, без підказки (`prompt-auto`)

Планувальник сам обере тип гілки з промпта + issue.

```bash
python main.py --launch-mode prompt-auto --url "https://github.com/fastapi/fastapi/issues/1663" --no-hil --prompt-file prompts/testing/prompt_only_duplicate_uk.txt

python main.py --launch-mode prompt-auto --url "https://github.com/fastapi/fastapi/issues/10370" --no-hil --prompt "Це bug чи feature? Процитуй фразу з issue і запропонуй 2–3 labels."

python main.py --launch-mode prompt-auto --url "https://github.com/psf/requests/issues/6109" --no-hil --prompt "Класифікуй issue (bug/feature/question/documentation/duplicate) без зайвих припущень."
```

## Швидкий перегляд усіх шаблонів команд у консолі

```bash
python main.py --launch-examples
python main.py --launch-examples --url https://github.com/fastapi/fastapi/issues/1663
```

(Другий рядок підставить свій URL у приклади.)
