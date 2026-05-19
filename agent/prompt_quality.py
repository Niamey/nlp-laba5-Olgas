"""Валідація користувацького промпта: порожній, сміття, prompt injection, multi-intent."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

VALID_TASK_TYPES = ("duplicate", "code_area", "stale", "classify")

# ── Ключові слова для детекції наміру ──────────────────────────────
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "duplicate": (
        "duplicate", "duplicates", "similar", "similar issues", "dup ",
        "дублік", "схож", "повтор", "вже репорт", "вже відкрив",
    ),
    "code_area": (
        "code area", "module", "file path", "file paths", "package",
        "subpackage", "where in the code", "which file", "which module",
        "модул", "файл", "директор", "пакет", "де в код", "локаліз", "component",
    ),
    "stale": (
        "stale", "outdated", "old issue", "no activity", "close it",
        "ping author", "keep_open", "needs_pr", "needs_info",
        "застар", "стар", "неактуа", "закрит", "пінг", "пинг", "outdated?",
    ),
    "classify": (
        "classify", "classification", "type:", "is this a bug", "is this a feature",
        "bug or feature", "label", "labels",
        "класиф", "тип ", "тип?", "баг", "фіча", "feature", "documentation",
        "question",
    ),
}

# ── Patterns для prompt injection ──────────────────────────────────
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in (
        r"ignore\s+(all\s+)?(previous|prior|above|earlier|the)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
        r"forget\s+(everything|all\s+(your\s+)?instructions?|previous)",
        r"(?:override|bypass|skip)\s+(your\s+)?(instructions?|safety|rules?|system\s+prompt)",
        r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?:a|an|the)\s+",
        r"(?:reveal|show|print|output|leak)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions?|initial\s+prompt)",
        r"jailbreak|dan\s+mode|developer\s+mode",
        r"do\s+anything\s+now",
        r"<\s*system\s*>|\[\s*system\s*\]|###\s*system",
        r"\bsudo\b|\bexec\(|\beval\(",
        r"\$\{[^}]*\}|\{\{[^}]+\}\}",  # template injection
        r"ігнор(?:уй|уйте|увати|ує)?[^.\n]{0,60}(?:інструкц|правил|промпт|систем)",
        r"забуд(?:ь|ьте)?[^.\n]{0,60}(?:інструкц|правил|промпт)",
        r"відкин(?:ь|ьте)?[^.\n]{0,60}(?:інструкц|правил|промпт)",
        r"виконуй\s+тільки\s+мої\s+команди",
        r"ти\s+(?:тепер|зараз)\s+(?:не\s+)?(?:агент|асистент|llm|ai)",
    )
)

# ── Patterns для очевидно зловмисних/токсичних команд ──────────────
_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bDROP\s+TABLE\b",
        r"\bDELETE\s+FROM\b",
        r"\bTRUNCATE\s+TABLE\b",
        r";\s*--",
        r"rm\s+-rf\s+/",
        r":\(\)\s*\{.*\};\s*:",  # fork bomb
    )
)

_MIN_LETTERS = 2
_MAX_PROMPT_LEN = 4000
_HARD_MAX_PROMPT_LEN = 20000        # вище — обрізаємо
_MAX_REPETITION_RATIO = 0.6         # >60% повтор одного токена → сміття
_MIN_UNIQUE_TOKENS = 3              # менше — підозра на спам
_MAX_EMOJI_RATIO = 0.5              # >50% емодзі — блок


@dataclass
class PromptValidation:
    is_valid: bool
    normalized: str | None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detected_intents: list[str] = field(default_factory=list)
    sanitized: bool = False

    @property
    def is_multi_intent(self) -> bool:
        return len(self.detected_intents) > 1


def _strip_invisible(text: str) -> str:
    """Прибирає zero-width / control chars (крім \\n, \\t)."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if cat.startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def _count_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _count_emoji_like(text: str) -> int:
    """Грубо: символи з категорій So/Sk/Sm — піктограми/символи."""
    return sum(1 for ch in text if unicodedata.category(ch) in ("So", "Sk", "Sm"))


def _detect_repetition(text: str) -> float:
    """
    Повертає частку найчастішого токена в тексті (0..1).
    Велике значення = підозра на спам ('duplicate duplicate duplicate ...').
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    if len(tokens) < 4:
        return 0.0
    from collections import Counter
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(tokens)


def _looks_like_garbage(text: str) -> bool:
    if not text:
        return True
    if _count_letters(text) < _MIN_LETTERS:
        return True
    noisy = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    if len(text) >= 4 and noisy / len(text) > 0.7:
        return True
    stripped = text.strip()
    if stripped and all(ch in "?.!,;:-_*~+=/\\|<>()[]{}\"'`" for ch in stripped):
        return True
    return False


def _is_emoji_only(text: str) -> bool:
    """True, якщо >50% символів — емодзі/піктограми."""
    if not text:
        return False
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return False
    emoji_count = _count_emoji_like("".join(visible))
    return emoji_count / len(visible) > _MAX_EMOJI_RATIO


def _detect_injection(text: str) -> list[str]:
    """Повертає список спрацьованих injection-паттернів (текстові описи)."""
    hits: list[str] = []
    for pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0).strip()[:60])
    return hits


def _detect_dangerous(text: str) -> list[str]:
    hits: list[str] = []
    for pat in _DANGEROUS_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0).strip()[:60])
    return hits


def detect_intents(text: str) -> list[str]:
    """Повертає task_types, що згадуються в тексті (за ключовими словами)."""
    if not text:
        return []
    low = text.lower()
    found: list[str] = []
    for task_type, keywords in INTENT_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            found.append(task_type)
    return found


def validate_user_prompt(raw: str | None) -> PromptValidation:
    """
    Валідація промпта користувача. Багаторівневі перевірки:

    Hard-fail (is_valid=False):
      - порожній / лише пробіли / лише control chars
      - сміття (тільки пунктуація, <2 літер, >70% noisy)
      - тільки емодзі
      - prompt injection (ignore previous, jailbreak, …)
      - очевидно небезпечні команди (DROP TABLE, rm -rf, fork bomb)
      - спам-повтор одного слова (>60%)

    Sanitize (is_valid=True, sanitized=True):
      - обрізання понад HARD_MAX (20000 символів)
      - попередження про довгий (>MAX), multi-intent, відсутність intents
    """
    if raw is None or not str(raw).strip():
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=["Промпт порожній. Додайте хоча б одне речення з метою тріажу."],
        )

    cleaned = _strip_invisible(str(raw)).strip()

    if not cleaned:
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                "Промпт містив лише невидимі/контрольні символи. "
                "Введіть текст українською або англійською."
            ],
        )

    if _looks_like_garbage(cleaned):
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                "Промпт виглядає як сміття або одиничні символи (?, !, …). "
                "Сформулюйте мету тріажу одним реченням, напр.: "
                "'Знайди схожі issues, чи це дублікат?'"
            ],
        )

    if _is_emoji_only(cleaned):
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                "Промпт складається переважно з емодзі/символів. "
                "Сформулюйте мету словами, напр.: 'Знайди схожі issues'."
            ],
        )

    injection_hits = _detect_injection(cleaned)
    if injection_hits:
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                "Виявлено спробу prompt injection: "
                + ", ".join(f"«{h}»" for h in injection_hits[:3])
                + ". Промпт відхилено. Використайте триаж-інструкції без переписування системного промпта."
            ],
        )

    dangerous_hits = _detect_dangerous(cleaned)
    if dangerous_hits:
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                "Виявлено потенційно небезпечні команди: "
                + ", ".join(f"«{h}»" for h in dangerous_hits[:3])
                + ". Промпт відхилено."
            ],
        )

    rep = _detect_repetition(cleaned)
    if rep > _MAX_REPETITION_RATIO:
        return PromptValidation(
            is_valid=False,
            normalized=None,
            issues=[
                f"Промпт виглядає як спам-повтор ({int(rep * 100)}% — один токен). "
                "Сформулюйте інструкцію природною мовою."
            ],
        )

    tokens = re.findall(r"\w+", cleaned.lower())
    if 0 < len(tokens) < _MIN_UNIQUE_TOKENS and len(set(tokens)) < _MIN_UNIQUE_TOKENS:
        # дуже мало унікальних токенів — пропускаємо тільки якщо є чіткий intent
        if not detect_intents(cleaned):
            return PromptValidation(
                is_valid=False,
                normalized=None,
                issues=[
                    "Занадто короткий і неінформативний промпт. "
                    "Додайте дієслово/мету: 'знайди дублікати', 'класифікуй', 'локалізуй модуль'."
                ],
            )

    sanitized = False
    if len(cleaned) > _HARD_MAX_PROMPT_LEN:
        cleaned = cleaned[:_HARD_MAX_PROMPT_LEN] + "\n…[truncated]"
        sanitized = True

    warnings: list[str] = []
    if sanitized:
        warnings.append(
            f"Промпт обрізано до {_HARD_MAX_PROMPT_LEN} символів (захист від overflow контексту)."
        )
    elif len(cleaned) > _MAX_PROMPT_LEN:
        warnings.append(
            f"Промпт довгий ({len(cleaned)} символів > {_MAX_PROMPT_LEN}). "
            "LLM може взяти не все. Скоротіть або винесіть у --prompt-file."
        )

    intents = detect_intents(cleaned)
    if not intents:
        warnings.append(
            "Не вдалось чітко визначити намір (duplicate / code_area / stale / classify) "
            "за ключовими словами. Планувальник вирішить сам — переконайтесь, що результат той, що треба."
        )

    if len(intents) > 1:
        warnings.append(
            f"У промпті знайдено кілька намірів: {', '.join(intents)}. "
            "Планувальник обере один (найсильніший сигнал) і поясне в reasoning. "
            "Щоб виконати кілька задач — запустіть тріаж окремо для кожної."
        )

    return PromptValidation(
        is_valid=True,
        normalized=cleaned,
        warnings=warnings,
        detected_intents=intents,
        sanitized=sanitized,
    )


def format_validation_report(v: PromptValidation) -> str:
    """Готовий блок для друку в консоль."""
    lines: list[str] = []
    if v.detected_intents:
        lines.append(f"  Detected intents: {', '.join(v.detected_intents)}")
    if v.sanitized:
        lines.append("  ✂ Sanitized: текст обрізано")
    for w in v.warnings:
        lines.append(f"  ⚠ {w}")
    for i in v.issues:
        lines.append(f"  ✗ {i}")
    return "\n".join(lines)
