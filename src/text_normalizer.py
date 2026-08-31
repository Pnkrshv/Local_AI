"""
Нормализация текста юридических документов.

Особенно важно для:
- кодов специальностей: 10 .05.0 2 -> 10.05.02
- номеров приказов: № 14 58 -> № 1458
- лишних пробелов перед знаками препинания
"""

import re


# Ищет трехчастные числовые коды вида:
# 10 .05.0 2
# 11 .0 2 .0 9
# 56.04.02
# 40 .05.0 1
_CODE_RE = re.compile(
    r"(?<!\d)"
    r"(\d{1,3}(?:[ \t]+\d+)*)"
    r"[ \t]*\.[ \t]*"
    r"(\d{1,3}(?:[ \t]+\d+)*)"
    r"[ \t]*\.[ \t]*"
    r"(\d{1,3}(?:[ \t]+\d+)*)"
    r"(?!\d)"
)

_ORDER_NUMBER_RE = re.compile(
    r"(№[ \t]*)(\d+(?:[ \t]+\d+)+)"
)


def _compact_digits(value: str) -> str:
    """
    Убирает все пробелы внутри числовой части.

    Пример:
        "0 2" -> "02"
        "1 4 5 8" -> "1458"
    """
    return re.sub(r"\s+", "", value)


def normalize_legal_text(text: str) -> str:
    """
    Нормализует текст перед чанкингом/индексацией.
    """
    if not text:
        return ""

    # 1. Нормализация кодов специальностей и похожих числовых кодов
    def replace_code(match: re.Match) -> str:
        part1 = _compact_digits(match.group(1))
        part2 = _compact_digits(match.group(2))
        part3 = _compact_digits(match.group(3))
        return f"{part1}.{part2}.{part3}"

    text = _CODE_RE.sub(replace_code, text)

    # 2. Нормализация номеров приказов/документов после "№"
    text = _ORDER_NUMBER_RE.sub(
        lambda m: m.group(1) + _compact_digits(m.group(2)),
        text
    )

    # 3. Убираем пробелы перед знаками препинания
    # Пример: "слово ." -> "слово."
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)

    # 4. Схлопываем повторяющиеся пробелы и табы
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()