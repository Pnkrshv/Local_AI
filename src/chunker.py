"""
Юридический чанкинг документов.

Улучшено:
- очистка текста от служебного мусора;
- нормализация кодов специальностей;
- распознавание списков;
- подсчет основных пунктов и дополнительных подпунктов;
- сохранение метаданных списка в каждый чанк.
"""

import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# ОЧИСТКА И НОРМАЛИЗАЦИЯ ТЕКСТА
# ============================================================

SUP_RE = re.compile(r"<sup>\s*(\d+)\s*</sup>", re.IGNORECASE)

SPECIALTY_CODE_RE = re.compile(
    r"(?<!\d)"
    r"(\d{1,3}(?:\s+\d+)*)"
    r"\s*\.\s*"
    r"(\d{1,3}(?:\s+\d+)*)"
    r"\s*\.\s*"
    r"(\d{1,3}(?:\s+\d+)*)"
    r"(?!\d)"
)

NOISE_PATTERNS = [
    re.compile(r"(?m)^\s*\d{2}\.\d{2}\.\d{4}\s*$"),
    re.compile(r"(?m)^\s*Система ГАРАНТ\s*$", re.IGNORECASE),
    re.compile(r"(?m)^\s*КонсультантПлюс\s*$", re.IGNORECASE),
    re.compile(r"(?m)^\s*Страница\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"(?m)^\s*\d+\s*$"),
]


def _compact_digits(value: str) -> str:
    return re.sub(r"\s+", "", value)


def normalize_legal_text(text: str) -> str:
    """
    Нормализует юридический текст:
    - превращает <sup>1</sup> в .1;
    - убирает лишние пробелы внутри кодов специальностей;
    - схлопывает повторяющиеся пробелы.
    """
    if not text:
        return ""

    # <sup>1</sup> -> .1
    text = SUP_RE.sub(r".\1", text)

    # 10 .05.0 2 -> 10.05.02
    def replace_code(match: re.Match) -> str:
        part1 = _compact_digits(match.group(1))
        part2 = _compact_digits(match.group(2))
        part3 = _compact_digits(match.group(3))
        return f"{part1}.{part2}.{part3}"

    text = SPECIALTY_CODE_RE.sub(replace_code, text)

    # Убираем лишние пробелы/табы
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def clean_legal_text(text: str, source: str = "") -> str:
    """
    Очищает текст от служебного мусора и повторяющихся заголовков источника.
    """
    if not text:
        return ""

    text = normalize_legal_text(text)

    # Убираем markdown-артефакты
    text = text.replace("**", "")
    text = text.replace("__", "")

    # Убираем типичный мусор справочных систем
    for pattern in NOISE_PATTERNS:
        text = pattern.sub("", text)

    # Убираем повторяющиеся колонтитулы, похожие на имя файла
    if source:
        stem = Path(source).stem.strip()

        if len(stem) >= 10:
            prefix = re.escape(stem[:30])
            text = re.sub(
                rf"(?m)^\s*{prefix}[^\n]{{0,250}}$",
                "",
                text
            )

    # Схлопываем пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# ПОИСК ЮРИДИЧЕСКИХ ЗАГОЛОВКОВ
# ============================================================

def find_legal_headers(text: str) -> list[dict]:
    """
    Ищет в тексте структурные элементы НПА:
    Разделы, Главы, Статьи, Приложения, Пункты.
    """
    headers = []

    # 1. Статья, Глава, Раздел, Приложение
    p1 = re.compile(
        r"(?im)^\s*\**\s*"
        r"(Раздел|Глава|Статья|Приложение)\s+"
        r"([IVXLC0-9]+(?:[-.][IVXLC0-9]+)*)"
        r"\s*\**\s*"
        r"(?:[.:]|\s+)?"
        r"\s*\**\s*"
        r"(.*?)\**\s*$"
    )

    for m in p1.finditer(text):
        headers.append(
            {
                "kind": m.group(1).title(),
                "number": m.group(2),
                "title": m.group(3).strip().strip("*").strip(),
                "start": m.start(),
            }
        )

    # 2. Римские цифры вида I., II., III. -> Раздел
    p2 = re.compile(
        r"(?im)^\s*\**\s*"
        r"([IVXLC]+)\."
        r"\s*(.*?)\**\s*$"
    )

    for m in p2.finditer(text):
        headers.append(
            {
                "kind": "Раздел",
                "number": m.group(1).upper(),
                "title": m.group(2).strip().strip("*").strip(),
                "start": m.start(),
            }
        )

    # 3. Пункты вида 1., 2., 12.
    p3 = re.compile(
        r"(?im)^\s*\**\s*"
        r"([0-9]+)\."
        r"\s+(.{10,}?)\**\s*$"
    )

    for m in p3.finditer(text):
        headers.append(
            {
                "kind": "Пункт",
                "number": m.group(1),
                "title": m.group(2).strip().strip("*").strip(),
                "start": m.start(),
            }
        )

    headers.sort(key=lambda x: x["start"])

    unique_headers = []
    seen_starts = set()

    for h in headers:
        if h["start"] not in seen_starts:
            unique_headers.append(h)
            seen_starts.add(h["start"])

    return unique_headers


def build_section_string(headers: list[dict]) -> str:
    """
    Собирает строку раздела, обрезая слишком длинные названия.
    """
    parts = []

    for h in headers:
        s = f'{h["kind"]} {h["number"]}'

        if h["title"]:
            title = h["title"]

            if len(title) > 80:
                title = title[:80].rsplit(" ", 1)[0] + "..."

            s += f". {title}"

        parts.append(s)

    return " / ".join(parts)


def split_into_legal_segments(full_text: str) -> list[dict]:
    """
    Разбивает текст на сегменты по найденным заголовкам.
    """
    headers = find_legal_headers(full_text)

    segments = []
    current_headers = []
    current_start = 0

    for h in headers:
        if h["start"] > current_start:
            segments.append(
                {
                    "text": full_text[current_start:h["start"]],
                    "headers": current_headers.copy(),
                }
            )

        if h["kind"] == "Раздел":
            current_headers = [h]
        elif h["kind"] == "Глава":
            current_headers = [x for x in current_headers if x["kind"] == "Раздел"] + [h]
        elif h["kind"] == "Статья":
            current_headers = [x for x in current_headers if x["kind"] in ("Раздел", "Глава")] + [h]
        elif h["kind"] == "Пункт":
            current_headers = [x for x in current_headers if x["kind"] in ("Раздел", "Глава", "Статья")] + [h]
        elif h["kind"] == "Приложение":
            current_headers = [h]

        current_start = h["start"]

    segments.append(
        {
            "text": full_text[current_start:],
            "headers": current_headers.copy(),
        }
    )

    return [s for s in segments if s["text"].strip()]


# ============================================================
# РАСПОЗНАВАНИЕ И ПОДСЧЕТ СПИСКОВ
# ============================================================

PAREN_MARKER_RE = re.compile(
    r"(?m)^\s*(?P<marker>\d{1,4}(?:\.\d{1,3})*|[а-яё])\s*\)\s+(?P<body>.{5,})"
)

DOT_MARKER_RE = re.compile(
    r"(?m)^\s*(?P<marker>\d{1,4}(?:\.\d{1,3})*|[а-яё])\s*\.\s+(?P<body>.{5,})"
)

DASH_MARKER_RE = re.compile(
    r"(?m)^\s*[-–—•*]\s+(?P<body>.{5,})"
)

CODE_LINE_RE = re.compile(
    r"(?m)^\s*\d{2}\.\d{2}\.\d{2}\b"
)


def _remove_headers_for_count(text: str, headers: list[dict]) -> str:
    """
    Удаляет из текста заголовки, чтобы они не считались пунктами списка.
    """
    if not text:
        return ""

    # Раздел / Глава / Статья / Приложение
    text = re.sub(
        r"(?m)^\s*(?:Раздел|Глава|Статья|Приложение)\b.*$",
        "",
        text
    )

    # Римские заголовки вида II. Полномочия
    text = re.sub(
        r"(?m)^\s*[IVXLC]+\.\s+.*$",
        "",
        text
    )

    # Заголовок текущего пункта/статьи
    if headers:
        last = headers[-1]
        num = last.get("number", "")

        if num:
            text = re.sub(
                rf"(?m)^\s*{re.escape(num)}\s*[\.:].*$",
                "",
                text,
                count=1
            )

    return text


def _collect_markers(regex: re.Pattern, text: str) -> list[str]:
    return [m.group("marker").strip().lower() for m in regex.finditer(text)]


def _analyze_markers(markers: list[str], style_suffix: str) -> dict:
    """
    Анализирует маркеры списка.

    Логика:
    - 1), 2), 80) -> основные пункты;
    - 23.1), 38.1), 79.2) -> дополнительные подпункты;
    - если список фактически состоит из 1.1, 1.2, считаем их основными;
    - буквы а), б), в) считаем отдельным типом списка.
    """
    letters = set()
    integers = set()
    fractions = set()
    raw_large = []

    for marker in markers:
        marker = marker.strip().lower()

        if not marker:
            continue

        # Буквенные маркеры: а), б), в)
        if marker.isalpha():
            letters.add(marker)
            continue

        # Дробные маркеры: 23.1, 79.2
        if "." in marker:
            fractions.add(marker)
            continue

        # Числовые маркеры
        if marker.isdigit():
            n = int(marker)

            if n > 100:
                raw_large.append(marker)
            else:
                integers.add(n)

    supplemental = set(fractions)

    # Обработка артефактов типа 381 ) -> 38.1)
    # Это часто возникает при извлечении надстрочных индексов из PDF.
    if raw_large:
        raw_large_unique = sorted(set(raw_large))

        # Если таких артефактов немного, считаем их дополнительными подпунктами.
        # Если их много, вероятно это реальные номера 101, 102, 103...
        if len(raw_large_unique) <= 8 and integers:
            max_normal = max(integers)

            for raw in raw_large_unique:
                if len(raw) >= 3:
                    base = raw[:-1]
                    last = raw[-1]

                    if base.isdigit():
                        base_num = int(base)

                        if base_num in integers or base_num <= max_normal:
                            supplemental.add(f"{base_num}.{last}")
                        else:
                            integers.add(int(raw))
                    else:
                        integers.add(int(raw))
                else:
                    integers.add(int(raw))
        else:
            for raw in raw_large_unique:
                integers.add(int(raw))

    # Если список буквенный
    if letters and len(letters) >= max(3, len(integers)):
        primary_count = len(letters)
        supplemental_count = len(supplemental)

        return {
            "list_style": f"letter_{style_suffix}",
            "list_count": primary_count,
            "primary_count": primary_count,
            "supplemental_count": supplemental_count,
        }

    # Если список фактически дробный: 1.1, 1.2, 2.1...
    if len(supplemental) >= 5 and len(supplemental) > len(integers) * 0.4:
        primary_count = len(integers) + len(supplemental)

        return {
            "list_style": f"numeric_fractional_{style_suffix}",
            "list_count": primary_count,
            "primary_count": primary_count,
            "supplemental_count": 0,
        }

    # Обычный нумерованный список: 1, 2, 3...
    primary_count = len(integers) if integers else len(supplemental)
    supplemental_count = len(supplemental)

    return {
        "list_style": f"numeric_{style_suffix}",
        "list_count": primary_count,
        "primary_count": primary_count,
        "supplemental_count": supplemental_count,
    }


def extract_list_stats(text: str, headers: list[dict] | None = None) -> dict:
    """
    Определяет тип списка и количество пунктов в тексте.
    """
    result = {
        "list_style": "none",
        "list_count": 0,
        "primary_count": 0,
        "supplemental_count": 0,
    }

    if not text:
        return result

    cleaned = clean_legal_text(text)
    cleaned = _remove_headers_for_count(cleaned, headers or [])

    # 1. Маркеры вида 1) 2) 3)
    paren_markers = _collect_markers(PAREN_MARKER_RE, cleaned)

    if len(paren_markers) >= 3:
        return _analyze_markers(paren_markers, "paren")

    # 2. Маркеры вида 1. 2. 3.
    dot_markers = _collect_markers(DOT_MARKER_RE, cleaned)

    if len(dot_markers) >= 3:
        return _analyze_markers(dot_markers, "dot")

    # 3. Маркеры вида - / — / •
    dash_matches = DASH_MARKER_RE.findall(cleaned)

    if len(dash_matches) >= 3:
        return {
            "list_style": "dash",
            "list_count": len(dash_matches),
            "primary_count": len(dash_matches),
            "supplemental_count": 0,
        }

    # 4. Списки кодов специальностей: 09.05.01, 10.05.02...
    code_matches = CODE_LINE_RE.findall(cleaned)

    if len(code_matches) >= 3:
        return {
            "list_style": "code",
            "list_count": len(code_matches),
            "primary_count": len(code_matches),
            "supplemental_count": 0,
        }

    return result


# ============================================================
# ЧАНКИНГ ДОКУМЕНТА
# ============================================================

def chunk_document(pages: list[dict]) -> list[dict]:
    """
    Разбивает документ на чанки.
    Для каждого сегмента считает метаданные списка.
    """
    if not pages:
        return []

    source = pages[0]["source"]

    full_text = "\n\n".join([p["text"] for p in pages])
    full_text = clean_legal_text(full_text, source)

    segments = split_into_legal_segments(full_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    final_chunks = []

    for segment in segments:
        section_str = build_section_string(segment["headers"]) or "Без раздела"

        text = segment["text"].strip()

        if not text:
            continue

        # Считаем список по всему сегменту, а не по отдельному чанку
        list_stats = extract_list_stats(text, segment["headers"])

        metadata = {
            "source": source,
            "section": section_str,
            "list_style": list_stats["list_style"],
            "list_count": list_stats["list_count"],
            "primary_count": list_stats["primary_count"],
            "supplemental_count": list_stats["supplemental_count"],
        }

        if len(text) > CHUNK_SIZE:
            sub_texts = text_splitter.split_text(text)

            for sub in sub_texts:
                final_chunks.append(
                    {
                        "text": sub,
                        "metadata": metadata.copy(),
                    }
                )
        else:
            final_chunks.append(
                {
                    "text": text,
                    "metadata": metadata.copy(),
                }
            )

    return final_chunks