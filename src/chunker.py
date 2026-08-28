"""
Финальный чанкинг юридических текстов.
Понимает Статьи, Главы, Разделы (в т.ч. римские цифры) и Пункты (арабские цифры).
Исправлена обработка артефактов вида "**Статья 1** ."
"""

import re
import sys
from pathlib import Path
from text_normalizer import normalize_legal_text

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_FOLDER


def find_legal_headers(text: str) -> list[dict]:
    """
    Ищет в тексте структурные элементы НПА.
    """
    headers = []
    
    # 1. Статья, Глава, Раздел, Приложение
    # Исправлено: разрешаем звёздочки, пробелы, точки и двоеточия после номера
    p1 = re.compile(
        r'(?im)^\s*\**\s*'
        r'(Раздел|Глава|Статья|Приложение)\s+'
        r'([IVXLC0-9]+(?:[-.][IVXLC0-9]+)*)'  # Номер (арабские или римские)
        r'\s*\**\s*'                          # Возможные звёздочки и пробелы после номера
        r'(?:[.:]|\s+)?'                      # Точка или двоеточие (опционально)
        r'\s*\**\s*'                          # Снова возможные звёздочки/пробелы
        r'(.*?)\**\s*$'                       # Название
    )
    for m in p1.finditer(text):
        headers.append({
            "kind": m.group(1).title(),
            "number": m.group(2),
            "title": m.group(3).strip().strip('*').strip(),
            "start": m.start(),
        })
        
    # 2. Римские цифры (I., II., III.) - это Разделы
    # Исправлено: разрешаем звёздочки вокруг
    p2 = re.compile(
        r'(?im)^\s*\**\s*'
        r'([IVXLC]+)\.'
        r'\s*(.*?)\**\s*$'
    )
    for m in p2.finditer(text):
        headers.append({
            "kind": "Раздел",
            "number": m.group(1).upper(),
            "title": m.group(2).strip().strip('*').strip(),
            "start": m.start(),
        })
        
    # 3. Пункты (1., 2., 12.) - арабские цифры с точкой.
    p3 = re.compile(
        r'(?im)^\s*\**\s*'
        r'([0-9]+)\.'
        r'\s+(.{10,}?)\**\s*$'
    )
    for m in p3.finditer(text):
        headers.append({
            "kind": "Пункт",
            "number": m.group(1),
            "title": m.group(2).strip().strip('*').strip(),
            "start": m.start(),
        })

    # Сортируем по позиции в тексте
    headers.sort(key=lambda x: x["start"])
    
    # Убираем дубликаты
    unique_headers = []
    seen_starts = set()
    for h in headers:
        if h["start"] not in seen_starts:
            unique_headers.append(h)
            seen_starts.add(h["start"])
            
    return unique_headers


def build_section_string(headers: list[dict]) -> str:
    """Собирает строку раздела, обрезая длинные названия."""
    parts = []
    for h in headers:
        s = f'{h["kind"]} {h["number"]}'
        if h["title"]:
            title = h["title"]
            if len(title) > 80:
                title = title[:80].rsplit(' ', 1)[0] + "..."
            s += f'. {title}'
        parts.append(s)
    return " / ".join(parts)


def split_into_legal_segments(full_text: str) -> list[dict]:
    """Разбивает текст на сегменты по найденным заголовкам."""
    headers = find_legal_headers(full_text)

    segments = []
    current_headers = []
    current_start = 0

    for h in headers:
        if h["start"] > current_start:
            segments.append({
                "text": full_text[current_start:h["start"]],
                "headers": current_headers.copy(),
            })

        # Обновляем стек заголовков (иерархия)
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

    segments.append({
        "text": full_text[current_start:],
        "headers": current_headers.copy(),
    })

    return [s for s in segments if s["text"].strip()]


def chunk_document(pages: list[dict]) -> list[dict]:
    if not pages:
        return []

    source = pages[0]["source"]

    full_text = "\n\n".join([p["text"] for p in pages])
    full_text = normalize_legal_text(full_text)

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

        if len(text) > CHUNK_SIZE:
            sub_texts = text_splitter.split_text(text)
            for sub in sub_texts:
                final_chunks.append({
                    "text": sub,
                    "metadata": {"source": source, "section": section_str}
                })
        else:
            final_chunks.append({
                "text": text,
                "metadata": {"source": source, "section": section_str}
            })

    return final_chunks


if __name__ == "__main__":
    from extract_pdf import extract_all_pdfs

    print("Извлечение текста для теста чанкинга...")
    all_pages = extract_all_pdfs(PDF_FOLDER)

    if not all_pages:
        print("Нет данных для чанкинга.")
        sys.exit(0)

    docs_pages = {}
    for p in all_pages:
        docs_pages.setdefault(p["source"], []).append(p)

    total_chunks = 0

    for source, pages in docs_pages.items():
        print(f"\n=== Чанкинг документа: {source} ===")
        chunks = chunk_document(pages)
        total_chunks += len(chunks)
        print(f"  Получено фрагментов: {len(chunks)}")

        # Покажем первые 3 фрагмента, у которых есть юридический раздел
        shown_count = 0
        for chunk in chunks:
            if chunk["metadata"]["section"] != "Без раздела" and shown_count < 3:
                print(f"\n  Пример {shown_count + 1}:")
                print(f"  Раздел: {chunk['metadata']['section']}")
                print(f"  Текст (первые 200 символов): {chunk['text'][:200]}...")
                shown_count += 1

    print(f"\nВсего чанков по всем документам: {total_chunks}")