"""
Индексация юридических документов.

Пайплайн:
1. Извлечение текста из PDF
2. Извлечение текста из DOC/DOCX
3. Юридический чанкинг
4. Вычисление эмбеддингов
5. Запись в ChromaDB
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
    DOCS_FOLDER,
)

from extract_pdf import extract_all_pdfs
from extract_docs import extract_all_docs
from chunker import chunk_document


def main():
    print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    all_pages = []

    print(f"Извлечение PDF-документов из: {PDF_FOLDER}")
    pdf_pages = extract_all_pdfs(PDF_FOLDER)

    if pdf_pages:
        all_pages.extend(pdf_pages)

    print(f"Извлечение Word-документов из: {DOCS_FOLDER}")
    doc_pages = extract_all_docs(DOCS_FOLDER)

    if doc_pages:
        all_pages.extend(doc_pages)

    if not all_pages:
        print("Не удалось извлечь текст ни из одного документа.")
        return

    docs_pages = {}

    for p in all_pages:
        docs_pages.setdefault(p["source"], []).append(p)

    all_chunks = []

    for source, pages in docs_pages.items():
        chunks = chunk_document(pages)
        all_chunks.extend(chunks)
        print(f"  {source}: {len(chunks)} фрагментов")

    print(f"Всего фрагментов для индексации: {len(all_chunks)}")

    if not all_chunks:
        print("Нечего индексировать.")
        return

    texts = [EMBED_PASSAGE_PREFIX + c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    print("Вычисление эмбеддингов...")

    embeddings = embedder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    db_path = Path(DB_DIR)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    write_batch_size = 500

    for i in tqdm(range(0, len(all_chunks), write_batch_size), desc="Запись в БД"):
        end = min(i + write_batch_size, len(all_chunks))

        collection.add(
            ids=ids[i:end],
            documents=[c["text"] for c in all_chunks[i:end]],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )

    print("Индексация завершена.")
    print(f"Всего фрагментов в базе: {collection.count()}")


if __name__ == "__main__":
    main()