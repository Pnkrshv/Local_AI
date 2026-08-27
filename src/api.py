"""
FastAPI сервер для юридического RAG-ассистента (асинхронная версия).
Принимает вопрос, ищет контекст и стримит ответ от Ollama.
"""
import json
import asyncio
import httpx
from collections import defaultdict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer

# Импортируем настройки из config.py
from config import (
    EMBED_MODEL_NAME, EMBED_QUERY_PREFIX, DB_DIR, COLLECTION_NAME,
    RETRIEVAL_TOP_K, FINAL_TOP_K, CONTEXT_MAX_CHARS,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
)

# ============================================================
# 1. ИНИЦИАЛИЗАЦИЯ (Выполняется 1 раз при запуске сервера)
# ============================================================
print("🚀 Запуск API...")
print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print(f"Подключение к ChromaDB: {DB_DIR}")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ База загружена. Фрагментов: {collection.count()}\n")

# ============================================================
# 2. НАСТРОЙКА FASTAPI
# ============================================================
app = FastAPI(title="Legal RAG API")

# Разрешаем запросы с любых доменов (нужно для React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# ============================================================
# 3. ЛОГИКА ПОИСКА И ПРОМПТА
# ============================================================
def truncate_text(text, max_len=8000):
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    return (truncated[:last_space] + "...") if last_space != -1 else truncated

# Вспомогательная функция для запуска синхронного кода в отдельном потоке
async def run_in_thread(func, *args):
    """Запускает блокирующую функцию в отдельном потоке, чтобы не блокировать event loop."""
    return await asyncio.to_thread(func, *args)

async def build_prompt(question: str):
    """Ищет фрагменты и формирует промпт. Возвращает только промпт.
    Все блокирующие операции (эмбеддинги, ChromaDB) вынесены в отдельные потоки."""
    
    # 1. Вычисляем эмбеддинг вопроса (CPU-bound операция -> в поток)
    q_emb = await run_in_thread(
        lambda: embedder.encode([EMBED_QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    )
    
    # 2. Запрос к ChromaDB (блокирующий -> в поток)
    def _query_chroma():
        results = collection.query(
            query_embeddings=q_emb,
            n_results=RETRIEVAL_TOP_K,
            include=['documents', 'metadatas']
        )
        return results['documents'][0], results['metadatas'][0]
    
    retrieved_docs, retrieved_metas = await run_in_thread(_query_chroma)

    # Фильтрация
    filtered_pairs = []
    seen = set()
    for doc, meta in zip(retrieved_docs, retrieved_metas):
        doc_clean = doc.strip()
        if len(doc_clean) >= 50 and doc_clean not in seen:
            seen.add(doc_clean)
            filtered_pairs.append((doc_clean, meta))
    
    top_pairs = filtered_pairs[:FINAL_TOP_K]
    if not top_pairs:
        return "В предоставленных документах нет информации для ответа на этот вопрос."

    # Умная догрузка и склейка
    section_counts = defaultdict(int)
    for _, meta in top_pairs:
        section_counts[meta.get('section', 'Без раздела')] += 1
    sections_to_expand = {s for s, count in section_counts.items() if count >= 2}

    expanded_pairs = []
    processed_sections = set()
    for doc, meta in top_pairs:
        section = meta.get('section', 'Без раздела')
        if section in sections_to_expand and section not in processed_sections:
            try:
                # Запрос к ChromaDB -> в поток
                def _get_section(sec=section):
                    return collection.get(where={"section": sec})
                
                all_section_data = await run_in_thread(_get_section)
                if all_section_data and all_section_data['documents']:
                    sorted_items = sorted(
                        zip(all_section_data['ids'], all_section_data['documents']),
                        key=lambda x: int(x[0].split('_')[1]) if x[0].startswith('chunk_') else 0
                    )
                    full_text = "\n\n".join([text for _, text in sorted_items])
                    expanded_pairs.append((full_text, meta))
                    processed_sections.add(section)
            except Exception:
                expanded_pairs.append((doc, meta))
        elif section not in sections_to_expand:
            expanded_pairs.append((doc, meta))

    # Склейка
    sections_dict = defaultdict(list)
    order_of_sections = []
    for doc, meta in expanded_pairs:
        section = meta.get('section', 'Без раздела')
        if section not in sections_dict:
            order_of_sections.append(section)
        sections_dict[section].append(doc)

    merged_pairs = []
    for section in order_of_sections:
        merged_text = "\n\n".join(sections_dict[section])
        first_meta = next(m for d, m in expanded_pairs if m.get('section') == section)
        merged_pairs.append((merged_text, first_meta))

    # Формирование контекста
    context_parts = []
    total_len = 0
    for i, (doc, meta) in enumerate(merged_pairs):
        source = meta.get('source', 'неизвестно')
        section = meta.get('section', 'Без раздела')
        part = f"[{i+1}] Источник: {source}\nРаздел: {section}\nТекст:\n{truncate_text(doc, 60000)}"
        if total_len + len(part) > CONTEXT_MAX_CHARS:
            break
        context_parts.append(part)
        total_len += len(part)

    context = "\n\n---\n\n".join(context_parts)
    
    # ПРОМПТ: модель сама указывает источники в конце ответа
    prompt = f"""Ты — строгий и внимательный юридический ассистент. Твоя задача — отвечать на вопросы пользователя ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов нормативно-правовых актов.

## КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. **Никаких внутренних знаний**: Используй ТОЛЬКО информацию из раздела "Контекст". Если ответа нет в контексте, напиши: "В предоставленных документах нет информации для ответа на этот вопрос."

2. **Строгий ответ на вопрос**:
   - Отвечай ТОЛЬКО на то, о чем спрашивает пользователь
   - НЕ добавляй от себя рассуждения, комментарии, "важные нюансы" или "дополнительную информацию", если их нет в контексте
   - НЕ пиши вводных фраз вроде "Согласно предоставленным документам...", "На основе анализа...", "Стоит отметить, что..."
   - Начинай ответ сразу с сути

3. **Цитирование источников**:
   - НЕ указывай источник в квадратных скобках после каждого пункта или предложения
   - Все источники указывай ТОЛЬКО один раз в самом конце ответа в разделе "Использованные источники"

4. **Запрет на выдумки**: Никогда не придумывай номера статей, пунктов, дат или фактов. Если информация неполная или обрывочная, честно укажи это.

5. **Обработка противоречий**: Если разные источники содержат противоречивую информацию, укажи это явно: "Источник X указывает..., однако Источник Y указывает..."

6. **⚠️ КРИТИЧЕСКОЕ ПРАВИЛО ДЛЯ ДЛИННЫХ СПИСКОВ (СТРОГО СОБЛЮДАТЬ!)**:
   Перед выводом любого списка ОБЯЗАТЕЛЬНО посчитай количество пунктов в контексте.
   
   ЕСЛИ пунктов БОЛЬШЕ 10:
   - Выведи РОВНО первые 10 пунктов (не больше!)
   - Сразу после 10-го пункта ОБЯЗАТЕЛЬНО прекрати вывод списка
   - Напиши отдельной строкой: "Это не полный список (всего N пунктов). Полный перечень содержится в [название документа], [название статьи/раздела]."
   - НЕ продолжай выводить оставшиеся пункты
   
   ЕСЛИ пунктов 10 ИЛИ МЕНЬШЕ:
   - Выведи все пункты полностью

7. **Стиль и форматирование**:
   - Формальный, но понятный язык
   - Используй Markdown для форматирования (заголовки, списки, жирный текст)
   - Приводи пункты полностью, как в тексте, но только те, которые есть в контексте

8. **Неясные вопросы**: Если вопрос сформулирован нечетко или требует уточнения, задай уточняющий вопрос вместо предположений.

## ФОРМАТ ИСТОЧНИКОВ В КОНЦЕ ОТВЕТА:

В самом конце ответа ОБЯЗАТЕЛЬНО добавь раздел:

Использованные источники:
1. [Полное название документа], [Раздел/Статья/Пункт]
2. [Полное название документа], [Раздел/Статья/Пункт]


## ПРИМЕР 1 — КОРОТКИЙ СПИСОК (менее 10 пунктов):

Вопрос: "Какие обязанности имеет работодатель по охране труда?"

Ответ:
Работодатель обязан обеспечить безопасные условия труда для всех сотрудников.

**Основные обязанности:**

- Проведение инструктажей по охране труда
- Организация медицинских осмотров работников
- Выдача средств индивидуальной защиты
- Обеспечение хранения и учета средств индивидуальной защиты
- Организация контроля за состоянием условий труда

Использованные источники:
Трудовой кодекс РФ, Глава 34, Статья 212


## ПРИМЕР 2 — ДЛИННЫЙ СПИСОК (более 10 пунктов):

Вопрос: "Какие полномочия у ФСО?"

Ответ:
Федеральная служба охраны (ФСО) России обладает следующими полномочиями:

**Основные полномочия (первые 10 из 80+):**

1. Осуществляет персональную охрану Президента Российской Федерации
2. Осуществляет транспортное обслуживание объектов государственной охраны
3. Обеспечивает Президента Российской Федерации специальной связью
4. Обеспечивает предоставление государственной охраны Президенту, прекратившему исполнение полномочий
5. Обеспечивает санитарно-эпидемиологическое благополучие объектов государственной охраны
6. Организует и проводит охранные, режимные, технические мероприятия на охраняемых объектах
7. Участвует в решении организационных вопросов, связанных с медицинским обслуживанием объектов государственной охраны
8. Разрабатывает и осуществляет специальные мероприятия по обеспечению безопасности объектов государственной охраны
9. Обеспечивает в установленном порядке специальной связью объекты государственной охраны
10. Разрабатывает и осуществляет меры, связанные с допуском лиц к работе по обслуживанию объектов государственной охраны

Это не полный список (всего 80+ пунктов). Полный перечень содержится в Указе Президента РФ от 7 августа 2004 г. N 1013 "Вопросы Федеральной службы охраны Российской Федерации", Раздел I. Общие положения, Пункт 12.

Использованные источники:
Указ Президента РФ от 7 августа 2004 г. N 1013 "Вопросы Федеральной службы охраны Российской Федерации", Раздел I. Общие положения, Пункт 12


## КОНТЕКСТ ИЗ ДОКУМЕНТОВ:

{context}

## ВОПРОС ПОЛЬЗОВАТЕЛЯ:

{question}

## ТВОЙ ОТВЕТ:"""
    return prompt

# ============================================================
# 4. СТРИМИНГ ОТ OLLAMA (ПОЛНОСТЬЮ АСИНХРОННЫЙ)
# ============================================================
async def stream_ollama(prompt: str):
    """Асинхронно делает запрос к Ollama и возвращает генератор кусочков текста.
    Использует httpx вместо requests для неблокирующего I/O."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0, "top_p": 0.9,
            "num_ctx": 32768, "num_predict": 16384, "repeat_penalty": 1.05
        }
    }
    
    # httpx.AsyncClient позволяет делать неблокирующие запросы
    # timeout=None, потому что у нас большой таймаут на уровне Ollama
    async with httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            # Асинхронно читаем поток построчно
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    json_data = json.loads(line)
                    if "response" in json_data:
                        yield json_data["response"]
                    if json_data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

# ============================================================
# 5. ЭНДПОИНТ (ТОЧКА ВХОДА)
# ============================================================
@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Готовим промпт (асинхронно, с выносом тяжелых операций в потоки)
    prompt = await build_prompt(request.question)
    
    # 2. Асинхронный генератор, который просто передаёт ответ модели "как есть"
    async def generate_response():
        async for chunk in stream_ollama(prompt):
            yield chunk

    # 3. Возвращаем асинхронный поток текста
    return StreamingResponse(generate_response(), media_type="text/plain; charset=utf-8")

# Точка входа, если запускать напрямую
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)