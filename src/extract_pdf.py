import os
import pymupdf4llm

from config import PDF_FOLDER


def extract_text_from_pdf(filepath: str) -> list[dict]:
    """
    Извлекает текст из одного PDF-файла по страницам.

    Args:
        filepath: путь к PDF-файлу.

    Returns:
        Список словарей вида:
        [
            {"page": 1, "text": "markdown текст страницы 1"},
            {"page": 2, "text": "markdown текст страницы 2"},
            ...
        ]
    """
    print(f"  Извлечение текста из: {os.path.basename(filepath)}")
    
    try:
        # page_chunks=True разбивает результат по страницам
        # write_images=False, чтобы не сохранять картинки (они нам пока не нужны)
        md_text_pages = pymupdf4llm.to_markdown(
            filepath,
            page_chunks=True,
            write_images=False,
        )
        
        pages = []
        for i, page_data in enumerate(md_text_pages):
            text = page_data.get("text", "")
            if text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text.strip()
                })
                
        return pages
        
    except Exception as e:
        print(f"  Ошибка при извлечении текста из {filepath}: {e}")
        return []


def extract_all_pdfs(pdf_folder: str) -> list[dict]:
    """
    Обрабатывает все PDF-файлы в папке.

    Returns:
        Список словарей вида:
        [
            {
                "source": "filename.pdf",
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """
    all_pages = []
    
    if not os.path.isdir(pdf_folder):
        print(f"Папка {pdf_folder} не найдена.")
        return all_pages
        
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("В папке нет PDF-файлов.")
        return all_pages
        
    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        pages = extract_text_from_pdf(filepath)
        
        for page in pages:
            page["source"] = filename
            all_pages.append(page)
            
    print(f"Всего извлечено страниц: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    # Простой тест
    pages = extract_all_pdfs(PDF_FOLDER)
    if pages:
        print("\n=== Пример первой страницы первого документа ===")
        print(f"Источник: {pages[0]['source']}")
        print(f"Страница: {pages[0]['page']}")
        print(f"Текст (первые 500 символов):\n{pages[0]['text'][:500]}...")
    else:
        print("Не удалось извлечь текст. Проверь, есть ли PDF-файлы в папке pdfs/")