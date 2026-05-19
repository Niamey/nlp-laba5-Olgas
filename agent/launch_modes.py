"""Чотири режими запуску одного тріажу: auto | hint-only | hint-prompt | prompt-auto."""
from __future__ import annotations

from pathlib import Path

VALID_HINTS = ("duplicate", "code_area", "stale", "classify")
LAUNCH_MODES = ("auto", "hint-only", "hint-prompt", "prompt-auto")


def normalize_hint(hint: str | None) -> str | None:
    if not hint or not str(hint).strip():
        return None
    h = str(hint).strip().lower()
    if h not in VALID_HINTS:
        raise ValueError(
            f"Невідомий --hint '{hint}'. Дозволено: {', '.join(VALID_HINTS)}"
        )
    return h


def resolve_launch(
    launch_mode: str | None,
    hint: str | None,
    user_prompt: str | None,
) -> tuple[str, str | None, str | None]:
    """
    Повертає (mode, task_hint, user_prompt) після валідації.

    launch_mode:
      auto        — без --hint і без промпту (тип обере plan_node лише з issue)
      hint-only   — лише --hint
      hint-prompt — --hint + --prompt або --prompt-file
      prompt-auto — лише --prompt (без --hint); plan_node обирає тип з промпта + issue
    """
    has_hint = bool(hint and str(hint).strip())
    has_prompt = bool(user_prompt and str(user_prompt).strip())

    if launch_mode:
        mode = launch_mode.strip().lower()
        if mode not in LAUNCH_MODES:
            raise ValueError(
                f"Невідомий --launch-mode '{launch_mode}'. Дозволено: {', '.join(LAUNCH_MODES)}"
            )
    elif has_hint and has_prompt:
        mode = "hint-prompt"
    elif has_hint:
        mode = "hint-only"
    elif has_prompt:
        mode = "prompt-auto"
    else:
        mode = "auto"

    norm_hint = normalize_hint(hint) if hint else None

    if mode == "auto":
        if norm_hint:
            raise ValueError(
                "Режим auto: не передавайте --hint. Або вкажіть --launch-mode hint-only / hint-prompt."
            )
        if has_prompt:
            raise ValueError(
                "Режим auto: не передавайте --prompt / --prompt-file. "
                "Для промпта без hint використайте --launch-mode prompt-auto."
            )
        return mode, None, None

    if mode == "hint-only":
        if not norm_hint:
            raise ValueError(
                f"Режим hint-only: обов'язковий --hint ({' | '.join(VALID_HINTS)})."
            )
        if has_prompt:
            raise ValueError(
                "Режим hint-only: заборонено --prompt і --prompt-file. "
                "Для промпту використайте --launch-mode hint-prompt."
            )
        return mode, norm_hint, None

    if mode == "prompt-auto":
        if norm_hint:
            raise ValueError(
                "Режим prompt-auto: не передавайте --hint. "
                "Тип обере планувальник з вашого промпта + issue. "
                "Якщо хочете явний hint — використайте --launch-mode hint-prompt."
            )
        if not has_prompt:
            raise ValueError(
                "Режим prompt-auto: потрібен --prompt \"...\" або --prompt-file PATH."
            )
        return mode, None, user_prompt.strip()

    # hint-prompt
    if not norm_hint:
        raise ValueError(
            f"Режим hint-prompt: обов'язковий --hint ({' | '.join(VALID_HINTS)})."
        )
    if not has_prompt:
        raise ValueError(
            "Режим hint-prompt: потрібен --prompt \"...\" або --prompt-file PATH."
        )
    return mode, norm_hint, user_prompt.strip()


def default_prompt_for_hint(hint: str) -> str:
    """Приклад промпту для демо (один режим = один hint)."""
    examples = {
        "duplicate": (
            "Зроби щонайменше 2 пошукові запити через search_similar_issues. "
            "Порівняй root cause, не лише заголовок. У звіті вкажи номери issues і вердикт duplicate/related/not."
        ),
        "code_area": (
            "Визнач primary module і 1–2 file paths лише якщо вони є в tool results або issue body. "
            "Не вигадуй шляхи. Дай confidence і коротке обґрунтування."
        ),
        "stale": (
            "Проаналізуй дати created/updated і останні коментарі. "
            "Дай конкретну recommended_action (close, ping_author, keep_open тощо)."
        ),
        "classify": (
            "У JSON і в звіті type має бути одним із: bug, feature, question, documentation, duplicate. "
            "У justification процитуй фразу з issue body. Запропонуй 2–3 labels."
        ),
    }
    return examples.get(hint, "Дотримуйся grounding: лише факти з tool results.")


def print_launch_examples(issue_url: str = "https://github.com/owner/repo/issues/123") -> None:
    """Друкує режими запуску та додаткові приклади для тестування."""
    url = issue_url
    sep = "═" * 62
    fast_dup = "https://github.com/fastapi/fastapi/issues/1663"
    fast_cls = "https://github.com/fastapi/fastapi/issues/10370"
    fast_code = "https://github.com/fastapi/fastapi/issues/5920"
    flask_stale = "https://github.com/pallets/flask/issues/4179"
    req_feat = "https://github.com/psf/requests/issues/6109"

    print(f"\n{sep}")
    print("  РЕЖИМИ ЗАПУСКУ (один issue URL)")
    print(sep)
    print(
        "\n  Режими (--launch-mode):\n"
        "    auto         — без підказки й без промпту (тип обере планувальник з issue)\n"
        "    hint-only    — лише --hint (duplicate | code_area | stale | classify)\n"
        "    hint-prompt  — --hint + ваш --prompt (або --prompt-file)\n"
        "    prompt-auto  — лише --prompt; планувальник обирає тип з промпта + issue\n"
    )
    print("  Підказки (--hint) для всіх можливостей:")
    for h in VALID_HINTS:
        print(f"    • {h}")

    print(f"\n{sep}")
    print("  1) AUTO — без підказки, без промпту")
    print(sep)
    print(
        f'  python main.py --launch-mode auto --url "{url}" --no-hil\n'
        "  → plan_node сам обере duplicate | code_area | stale | classify.\n"
    )

    print(sep)
    print("  2) HINT-ONLY — лише підказка (без промпту)")
    print(sep)
    for h in VALID_HINTS:
        print(f'\n  # режим: {h}')
        print(f'  python main.py --launch-mode hint-only --hint {h} --url "{url}" --no-hil')

    print(f"\n{sep}")
    print("  3) HINT-PROMPT — підказка + ваш промпт (для кожної можливості)")
    print(sep)
    print("  Приклад для duplicate:")
    dup_prompt = default_prompt_for_hint("duplicate").replace('"', '\\"')
    print(
        f'  python main.py --launch-mode hint-prompt --hint duplicate --url "{url}" --no-hil '
        f'--prompt "{dup_prompt}"'
    )
    print("\n  Або файл prompts/classify.txt:")
    print(
        f'  python main.py --launch-mode hint-prompt --hint classify --url "{url}" '
        f"--prompt-file prompts/classify.txt --no-hil"
    )

    print(f"\n{sep}")
    print("  4) PROMPT-AUTO — лише ваш промпт, тип обере планувальник")
    print(sep)
    print(
        f'  python main.py --launch-mode prompt-auto --url "{url}" --no-hil '
        f'--prompt "Знайди схожі issues і скажи, чи це дублікат"'
    )
    print(
        f'\n  python main.py --launch-mode prompt-auto --url "{url}" --no-hil '
        f'--prompt "Це класифікація: bug чи feature? Запропонуй labels."'
    )
    print(
        f'\n  python main.py --launch-mode prompt-auto --url "{url}" '
        f"--prompt-file prompts/duplicate.txt --no-hil"
    )
    print(f"\n{sep}")
    print("  5) ПРИКЛАДИ ДЛЯ ТЕСТУ (prompt + hint та prompt без hint)")
    print(sep)
    print(
        "  Повний список: prompts/TESTING_EXAMPLES.md\n"
        "\n  hint-prompt (підказка + промпт, файл):\n"
        f'  python main.py --launch-mode hint-prompt --hint duplicate --url "{fast_dup}" '
        f"--no-hil --prompt-file prompts/testing/hint_duplicate_uk.txt\n\n"
        f'  python main.py --launch-mode hint-prompt --hint classify --url "{fast_cls}" '
        f"--no-hil --prompt-file prompts/classify.txt\n\n"
        f'  python main.py --launch-mode hint-prompt --hint code_area --url "{fast_code}" '
        f"--no-hil --prompt-file prompts/code_area.txt\n\n"
        f'  python main.py --launch-mode hint-prompt --hint stale --url "{flask_stale}" '
        f"--no-hil --prompt-file prompts/stale.txt\n"
    )
    print(
        "  hint-prompt (інлайн-промпт):\n"
        f'  python main.py --launch-mode hint-prompt --hint duplicate --url "{fast_dup}" '
        f'--no-hil --prompt "Зроби два пошуки схожих issues; у звіті номери та вердикт."\n'
    )
    print(
        "  prompt-auto (лише промпт, тип гілки обере планувальник):\n"
        f'  python main.py --launch-mode prompt-auto --url "{fast_dup}" --no-hil '
        f"--prompt-file prompts/testing/prompt_only_duplicate_uk.txt\n\n"
        f'  python main.py --launch-mode prompt-auto --url "{fast_cls}" --no-hil '
        f'--prompt "Це bug чи feature? Процитуй фразу з issue і запропонуй 2–3 labels."\n\n'
        f'  python main.py --launch-mode prompt-auto --url "{req_feat}" --no-hil '
        f'--prompt "Класифікуй issue; не вигадуй фактів поза tool results."\n'
    )
    print(f"{sep}\n")
