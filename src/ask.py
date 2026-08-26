"""
Скрипт для вопросов и ответов (RAG).

Использует проиндексированную базу ChromaDB,
находит релевантные фрагменты и передает их в Ollama (Qwen2.5:14b).
Добавлена умная догрузка полных разделов.
"""

import sys
from pathlib import Path
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
import requests

sys.path.append(str(Path(__file__).parent))
from config import (
    EMBED_MODEL_NAME,
    EMBED_QUERY_PREFIX,
    DB_DIR,
    COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    FINAL_TOP_K,
    CONTEXT_MAX_CHARS,
    OLLAMA_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)


def truncate_text(text, max_len=8000):
    """Обрезает текст, не разрезая слова."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space != -1:
        truncated = truncated[:last_space]
    return truncated + "..."


def main():
    print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Подключение к базе данных: {DB_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except ValueError:
        print(f"Ошибка: коллекция '{COLLECTION_NAME}' не найдена.")
        print("Сначала запустите python src/build_index.py")
        return

    print(f"Коллекция загружена. Фрагментов в базе: {collection.count()}")
    print(f"Модель для генерации: {OLLAMA_MODEL}")
    print("\n" + "="*60)
    print("Система готова! Задавайте вопросы по загруженным НПА.")
    print("Для выхода введите 'exit' или 'quit'")
    print("="*60)

    while True:
        question = input("\n❓ Ваш вопрос: ").strip()
        if question.lower() in ('exit', 'quit', 'выход'):
            break
        if not question:
            continue

        # 1. Поиск фрагментов
        print("\n🔎 Поиск релевантных фрагментов...")
        q_emb = embedder.encode(
            [EMBED_QUERY_PREFIX + question],
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=q_emb,
            n_results=RETRIEVAL_TOP_K,
            include=['documents', 'metadatas']
        )

        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]

        # 2. Фильтрация и отбор топ-N
        filtered_pairs = []
        seen = set()
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            doc_clean = doc.strip()
            if len(doc_clean) < 50:
                continue
            if doc_clean not in seen:
                seen.add(doc_clean)
                filtered_pairs.append((doc_clean, meta))

        # Берем FINAL_TOP_K лучших
        top_pairs = filtered_pairs[:FINAL_TOP_K]

        if not top_pairs:
            print("Не удалось найти подходящие фрагменты.")
            continue

        # ============================================================
        # 2.1. УМНАЯ ДОГРУЗКА: если раздел разбит на много кусков,
        # собираем его целиком из базы
        # ============================================================
        section_counts = defaultdict(int)
        for doc, meta in top_pairs:
            section_counts[meta.get('section', 'Без раздела')] += 1
        
        # Разделы, у которых больше 1 фрагмента в топе — кандидаты на догрузку
        sections_to_expand = {s for s, count in section_counts.items() if count >= 2}
        
        expanded_pairs = []
        processed_sections = set()
        
        for doc, meta in top_pairs:
            section = meta.get('section', 'Без раздела')
            
            if section in sections_to_expand and section not in processed_sections:
                try:
                    # Запрашиваем ВСЕ фрагменты этого раздела из ChromaDB
                    all_section_data = collection.get(where={"section": section})
                    
                    if all_section_data and all_section_data['documents']:
                        # Сортируем по ID (chunk_1, chunk_2...), чтобы сохранить порядок текста
                        sorted_items = sorted(
                            zip(all_section_data['ids'], all_section_data['documents']),
                            key=lambda x: int(x[0].split('_')[1]) if x[0].startswith('chunk_') else 0
                        )
                        full_section_text = "\n\n".join([text for _, text in sorted_items])
                        print(f"  📦 Догружен полный раздел: {section} ({len(sorted_items)} фрагментов)")
                        expanded_pairs.append((full_section_text, meta))
                        processed_sections.add(section)
                    else:
                        expanded_pairs.append((doc, meta))
                except Exception as e:
                    print(f"  ⚠️ Не удалось догрузить раздел {section}: {e}")
                    expanded_pairs.append((doc, meta))
            
            elif section not in sections_to_expand:
                expanded_pairs.append((doc, meta))
        
        top_pairs = expanded_pairs

        # ============================================================
        # 2.2. Склейка фрагментов с одинаковым разделом
        # ============================================================
        sections_dict = defaultdict(list)
        order_of_sections = []
        
        for doc, meta in top_pairs:
            section = meta.get('section', 'Без раздела')
            if section not in sections_dict:
                order_of_sections.append(section)
            sections_dict[section].append(doc)
        
        merged_pairs = []
        for section in order_of_sections:
            merged_text = "\n\n".join(sections_dict[section])
            first_meta = next(m for d, m in top_pairs if m.get('section') == section)
            merged_pairs.append((merged_text, first_meta))
        
        top_pairs = merged_pairs

        # 3. Формирование контекста
        context_parts = []
        sources_info = []
        total_len = 0

        for i, (doc, meta) in enumerate(top_pairs):
            source = meta.get('source', 'неизвестно')
            section = meta.get('section', 'Без раздела')
            
            # Обрезаем слишком длинные куски
            doc_truncated = truncate_text(doc, 60000)
            
            part = f"[{i+1}] Источник: {source}\nРаздел: {section}\nТекст:\n{doc_truncated}"
            
            if total_len + len(part) > CONTEXT_MAX_CHARS:
                break
                
            context_parts.append(part)
            total_len += len(part)
            
            sources_info.append(f"{i+1}. {source} | {section}")

        context = "\n\n---\n\n".join(context_parts)

        # 4. Формирование строгого промпта
        prompt = f"""Ты — строгий и внимательный юридический ассистент. Отвечай на вопрос пользователя ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов нормативно-правовых актов.

Правила:
1. Не используй свои внутренние знания. Если ответа нет в тексте фрагментов, так и напиши: "В предоставленных документах нет информации для ответа на этот вопрос".
2. Не выдумывай номера статей, пунктов или факты.
3. Если перечисляешь пункты или полномочия, приводи их полностью, как в тексте, но только те, которые есть в контексте.
4. ВАЖНО: если ты видишь, что список пунктов обрывается или явно неполный (например, после пункта 8 идёт "..." или логика нарушена), ОБЯЗАТЕЛЬНО напиши в начале ответа: "Внимание: в предоставленных фрагментах содержится только часть списка. Ниже приведены пункты, которые удалось найти."
5. В конце ответа обязательно укажи список использованных источников в формате:
   [Номер] Название документа, Раздел/Статья/Пункт.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

        # 5. Запрос к Ollama
        print("🤖 Генерация ответа (это может занять 60-90 секунд для больших контекстов)...")
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_ctx": 32768,         # Максимальный контекст для Qwen2.5
                "num_predict": 16384,      # Лимит на длину ответа
                "repeat_penalty": 1.05,
            }
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            answer = response.json().get('response', '').strip()

            print("\n" + "="*60)
            print("✅ ОТВЕТ:")
            print("="*60)
            print(answer)
            print("\n" + "-"*60)
            print("📚 Использованные источники:")
            for src in sources_info:
                print(f"  • {src}")
            print("-"*60)

        except requests.exceptions.ConnectionError:
            print("❌ Ошибка: не удалось подключиться к Ollama. Убедись, что она запущена.")
        except requests.exceptions.Timeout:
            print("❌ Ошибка: Ollama не ответила вовремя (таймаут).")
        except Exception as e:
            print(f"❌ Ошибка при обращении к Ollama: {e}")


if __name__ == "__main__":
    main()