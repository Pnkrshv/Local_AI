"""
Индексация юридических документов.

Пайплайн:
1. Извлечение текста из PDF (extract_pdf)
2. Юридический чанкинг (chunker)
3. Вычисление эмбеддингов с префиксом "passage:"
4. Запись в ChromaDB с метаданными (source, section)
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from config import (
    EMBED_MODEL_NAME,
    EMBED_PASSAGE_PREFIX,
    DB_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    PDF_FOLDER,
)
from extract_pdf import extract_all_pdfs
from chunker import chunk_document


def main():
    # ============================================================
    # 1. Загрузка модели эмбеддингов
    # ============================================================
    print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
    print("При первом запуске модель скачивается (~2 ГБ), нужен интернет.\n")
    
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Устройство для эмбеддингов: {embedder.device}")
    if str(embedder.device) == "cpu":
        print("⚠️  Внимание: используется CPU. Если есть GPU, эмбеддинги будут медленнее.")
    else:
        print("✅ Используется GPU — эмбеддинги будут быстрыми.\n")

    # ============================================================
    # 2. Извлечение и чанкинг всех документов
    # ============================================================
    print(f"Извлечение и чанкинг документов из: {PDF_FOLDER}\n")
    all_pages = extract_all_pdfs(PDF_FOLDER)

    if not all_pages:
        print("Не удалось извлечь текст. Проверь папку pdfs/")
        return

    # Группируем страницы по документам
    docs_pages = {}
    for p in all_pages:
        docs_pages.setdefault(p["source"], []).append(p)

    all_chunks = []
    for source, pages in docs_pages.items():
        chunks = chunk_document(pages)
        all_chunks.extend(chunks)
        print(f"  {source}: {len(chunks)} фрагментов")

    print(f"\nВсего фрагментов для индексации: {len(all_chunks)}")

    if not all_chunks:
        print("Нечего индексировать.")
        return

    # ============================================================
    # 3. Подготовка текстов с префиксом "passage:"
    # ============================================================
    # ВАЖНО: для моделей multilingual-e5 нужен префикс "passage: "
    texts = [EMBED_PASSAGE_PREFIX + c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # ============================================================
    # 4. Вычисление эмбеддингов
    # ============================================================
    print("\nВычисление эмбеддингов (может занять некоторое время)...")
    embeddings = embedder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ============================================================
    # 5. Запись в ChromaDB
    # ============================================================
    db_path = Path(DB_DIR)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nСохранение в ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))

    # Удаляем старую коллекцию, если она есть
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Записываем батчами (у Chroma есть лимит на размер одного запроса)
    write_batch_size = 500
    for i in tqdm(range(0, len(all_chunks), write_batch_size), desc="Запись в БД"):
        end = min(i + write_batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end],
            documents=[c["text"] for c in all_chunks[i:end]],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )

    print(f"\n✅ Индексация завершена!")
    print(f"Всего фрагментов в базе: {collection.count()}")
    print(f"База сохранена в: {db_path}")


if __name__ == "__main__":
    main()