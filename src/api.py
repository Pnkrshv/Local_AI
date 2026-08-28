"""
FastAPI сервер для юридического RAG-ассистента.

Реализовано:
- гибридный поиск: векторный поиск + BM25;
- Reranker для точной пересортировки;
- Metadata Boosting для документов-справочников;
- безопасная склейка разделов только внутри одного источника;
- чистый вывод в консоль (только факт запуска);
- умная обработка неполных списков в промпте.
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import asyncio
import re
import logging

import httpx

from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

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
    RERANKER_MODEL_NAME,
)


# ============================================================
# Подавление логов
# ============================================================

logging.basicConfig(level=logging.WARNING)

for _logger_name in (
    "chromadb",
    "sentence_transformers",
    "transformers",
    "httpx",
    "httpcore",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


# ============================================================
# 1. ИНИЦИАЛИЗАЦИЯ
# ============================================================

embedder = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(COLLECTION_NAME)


# ============================================================
# 2. BM25-ИНДЕКС
# ============================================================

BM25_TOKEN_RE = re.compile(
    r"[а-яёa-z0-9]+(?:[./-][а-яёa-z0-9]+)*",
    re.IGNORECASE
)


def tokenize_for_bm25(text: str):
    if not text:
        return []
    return BM25_TOKEN_RE.findall(text.lower())


_all_data = collection.get(include=["documents", "metadatas"])

BM25_IDS = _all_data.get("ids") or []
BM25_DOCS = _all_data.get("documents") or []
BM25_METAS = _all_data.get("metadatas") or []

BM25_INDEX = None

if BM25Okapi is not None and BM25_DOCS:
    tokenized_docs = []
    for doc in BM25_DOCS:
        tokens = tokenize_for_bm25(doc)
        if not tokens:
            tokens = ["empty"]
        tokenized_docs.append(tokens)
    BM25_INDEX = BM25Okapi(tokenized_docs)


# ============================================================
# 3. FASTAPI
# ============================================================

app = FastAPI(title="Legal RAG API")

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
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def truncate_text(text: str, max_len: int = 8000) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space != -1:
        return truncated[:last_space] + "..."
    return truncated + "..."


async def run_in_thread(func, *args):
    return await asyncio.to_thread(func, *args)


def _get_first(result: dict, key: str):
    value = result.get(key)
    if not value:
        return []
    if isinstance(value, list):
        return value[0] if value else []
    return []


def _chunk_sort_key(item):
    chunk_id, _ = item
    try:
        if isinstance(chunk_id, str) and chunk_id.startswith("chunk_"):
            return int(chunk_id.split("_")[1])
    except Exception:
        pass
    return 0


# ============================================================
# 5. ГИБРИДНЫЙ ПОИСК: VECTOR + BM25 + RRF
# ============================================================

def retrieve_hybrid(
    question: str,
    q_emb,
    vector_top_k: int,
    bm25_top_k: int,
    rrf_k: int = 60,
):
    total = collection.count()
    if total == 0:
        return [], []

    vector_top_k = max(1, min(vector_top_k, total))

    vector_res = collection.query(
        query_embeddings=q_emb,
        n_results=vector_top_k,
        include=["documents", "metadatas"],
    )

    v_ids = _get_first(vector_res, "ids")
    v_docs = _get_first(vector_res, "documents")
    v_metas = _get_first(vector_res, "metadatas")

    scores = {}
    docs_by_id = {}
    metas_by_id = {}

    for rank, (chunk_id, doc, meta) in enumerate(zip(v_ids, v_docs, v_metas), start=1):
        if not chunk_id:
            continue
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        docs_by_id[chunk_id] = doc or ""
        metas_by_id[chunk_id] = meta or {}

    if BM25_INDEX is not None and BM25_IDS:
        q_tokens = tokenize_for_bm25(question)
        if q_tokens:
            bm25_scores = BM25_INDEX.get_scores(q_tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:bm25_top_k]

            for rank, idx in enumerate(top_indices, start=1):
                if bm25_scores[idx] <= 0.0:
                    break
                if idx >= len(BM25_IDS):
                    continue
                chunk_id = BM25_IDS[idx]
                if not chunk_id:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                docs_by_id[chunk_id] = BM25_DOCS[idx] if idx < len(BM25_DOCS) else ""
                metas_by_id[chunk_id] = BM25_METAS[idx] if idx < len(BM25_METAS) else {}

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return (
        [docs_by_id[id_] for id_ in ranked_ids],
        [metas_by_id[id_] for id_ in ranked_ids]
    )


# ============================================================
# 6. ЛОГИКА ПОИСКА И ПРОМПТА
# ============================================================

async def build_prompt(question: str) -> str:
    q_emb = await run_in_thread(
        lambda: embedder.encode(
            [EMBED_QUERY_PREFIX + question],
            normalize_embeddings=True
        ).tolist()
    )

    def _query_hybrid():
        return retrieve_hybrid(
            question=question,
            q_emb=q_emb,
            vector_top_k=RETRIEVAL_TOP_K,
            bm25_top_k=RETRIEVAL_TOP_K,
        )

    retrieved_docs, retrieved_metas = await run_in_thread(_query_hybrid)

    def _rerank():
        if not retrieved_docs:
            return [], []
        pairs = [[question, doc] for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        scored_items = list(zip(retrieved_docs, retrieved_metas, scores))
        scored_items.sort(key=lambda x: x[2], reverse=True)
        sorted_docs = [item[0] for item in scored_items]
        sorted_metas = [item[1] for item in scored_items]
        return sorted_docs, sorted_metas

    retrieved_docs, retrieved_metas = await run_in_thread(_rerank)

    def is_reference_doc(source: str) -> bool:
        source_lower = source.lower()
        keywords = ["фгос", "стандарт", "специальност", "направлен", "перечень", "список", "классификат"]
        return any(kw in source_lower for kw in keywords)

    boosted_docs, boosted_metas = [], []
    regular_docs, regular_metas = [], []

    for doc, meta in zip(retrieved_docs, retrieved_metas):
        source = (meta or {}).get("source", "")
        if is_reference_doc(source):
            boosted_docs.append(doc)
            boosted_metas.append(meta)
        else:
            regular_docs.append(doc)
            regular_metas.append(meta)

    retrieved_docs = boosted_docs + regular_docs
    retrieved_metas = boosted_metas + regular_metas

    filtered_pairs = []
    seen = set()

    for doc, meta in zip(retrieved_docs, retrieved_metas):
        doc_clean = (doc or "").strip()
        meta = meta or {}
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        key = (source, section, doc_clean[:2000])

        if len(doc_clean) >= 50 and key not in seen:
            seen.add(key)
            filtered_pairs.append((doc_clean, meta))

    top_pairs = filtered_pairs[:FINAL_TOP_K]

    if not top_pairs:
        return (
            "Ты — юридический ассистент. "
            "В предоставленных документах нет информации для ответа на этот вопрос. "
            "Выведи ровно одну фразу: "
            "В предоставленных документах нет информации для ответа на этот вопрос."
        )

    section_counts = defaultdict(int)
    for _, meta in top_pairs:
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        if section == "Без раздела":
            continue
        section_counts[(source, section)] += 1

    sections_to_expand = {k for k, count in section_counts.items() if count >= 2}
    expanded_pairs = []
    processed_sections = set()

    for doc, meta in top_pairs:
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        section_key = (source, section)

        if section_key in sections_to_expand:
            if section_key in processed_sections:
                continue
            try:
                def _get_section(src=source, sec=section):
                    return collection.get(where={"source": src, "section": sec})

                all_section_data = await run_in_thread(_get_section)
                if all_section_data and all_section_data.get("documents"):
                    docs = all_section_data["documents"]
                    ids = all_section_data.get("ids") or []
                    if len(ids) != len(docs):
                        ids = [f"chunk_{i}" for i in range(len(docs))]
                    sorted_items = sorted(zip(ids, docs), key=_chunk_sort_key)
                    full_text = "\n\n".join([text for _, text in sorted_items])
                    expanded_pairs.append((full_text, meta))
                    processed_sections.add(section_key)
                else:
                    expanded_pairs.append((doc, meta))
                    processed_sections.add(section_key)
            except Exception:
                expanded_pairs.append((doc, meta))
                processed_sections.add(section_key)
        else:
            expanded_pairs.append((doc, meta))

    sections_dict = defaultdict(list)
    order_of_sections = []
    meta_by_section_key = {}

    for doc, meta in expanded_pairs:
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        section_key = (source, section)
        if section_key not in sections_dict:
            order_of_sections.append(section_key)
            meta_by_section_key[section_key] = meta
        sections_dict[section_key].append(doc)

    merged_pairs = []
    for section_key in order_of_sections:
        merged_text = "\n\n".join(sections_dict[section_key])
        merged_meta = meta_by_section_key[section_key]
        merged_pairs.append((merged_text, merged_meta))

    context_parts = []
    total_len = 0

    for i, (doc, meta) in enumerate(merged_pairs):
        source = meta.get("source", "неизвестно")
        section = meta.get("section", "Без раздела")
        part = (
            f"[{i + 1}] Источник: {source}\n"
            f"Раздел: {section}\n"
            f"Текст:\n"
            f"{truncate_text(doc, 60000)}"
        )
        if total_len + len(part) > CONTEXT_MAX_CHARS:
            break
        context_parts.append(part)
        total_len += len(part)

    if not context_parts:
        return (
            "Ты — юридический ассистент. "
            "В предоставленных документах нет информации для ответа на этот вопрос. "
            "Выведи ровно одну фразу: "
            "В предоставленных документах нет информации для ответа на этот вопрос."
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Ты — строгий и внимательный юридический ассистент. Твоя задача — отвечать на вопросы пользователя ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов нормативно-правовых актов.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. Никаких внутренних знаний: Используй ТОЛЬКО информацию из раздела "Контекст". Если ответа нет в контексте, напиши: "В предоставленных документах нет информации для ответа на этот вопрос."

2. Строгий ответ на вопрос:
- Отвечай ТОЛЬКО на то, о чем спрашивает пользователь
- НЕ добавляй от себя рассуждения, комментарии, "важные нюансы" или "дополнительную информацию", если их нет в контексте
- НЕ пиши вводных фраз вроде "Согласно предоставленным документам...", "На основе анализа...", "Стоит отметить, что..."
- Начинай ответ сразу с сути
- Не пиши источники, если информация не была найдена (ВАЖНО!)

3. Цитирование источников:
- НЕ указывай источник в квадратных скобках после каждого пункта или предложения
- Все источники указывай ТОЛЬКО один раз в самом конце ответа в разделе "Использованные источники"

4. Запрет на выдумки: Никогда не придумывай номера статей, пунктов, дат или фактов. Если информация неполная или обрывочная, честно укажи это.

5. Обработка противоречий: Если разные источники содержат противоречивую информацию, укажи это явно: "Источник X указывает..., однако Источник Y указывает..."

⚠️ КРИТИЧЕСКОЕ ПРАВИЛО ДЛЯ СПИСКОВ И ПЕРЕЧНЕЙ (СТРОГО СОБЛЮДАТЬ!):

Нормативные акты часто содержат длинные перечни (полномочия, права, обязанности, функции, специальности), которые могут состоять из десятков подпунктов. Из-за ограничений размера фрагмента в контекст могла попасть только часть такого перечня.

1. Если в контексте содержится БОЛЬШЕ 10 пунктов списка:
- Выведи РОВНО первые 10 пунктов (не больше!)
- Сразу после 10-го пункта ОБЯЗАТЕЛЬНО прекрати вывод списка
- Напиши отдельной строкой: "Это не полный список. Полный перечень содержится в [название документа], [название статьи/раздела]."
- НЕ продолжай выводить оставшиеся пункты

2. Если в контексте 10 ИЛИ МЕНЬШЕ пунктов, НО они выглядят как часть большого перечня (например, это подпункты конкретного пункта Указа, или список обрывается, или это перечень полномочий/прав/обязанностей/специальностей):
- Выведи все имеющиеся пункты полностью.
- ОБЯЗАТЕЛЬНО добавь в конце списка предупреждение:
"Примечание: В предоставленном фрагменте приведена часть пунктов. Полный перечень содержится в [название документа], [название статьи/раздела]."

3. Если список короткий (2-5 пунктов) и логически завершен (например, виды наказаний или формы обучения), предупреждение добавлять не нужно.

🚨 ПРИОРИТЕТ КОНКРЕТИКИ НАД "ВОДОЙ":
Если пользователь спрашивает "Какие есть стандарты?", "Какие специальности?", "Какие направления?", "Какие полномочия?", он ожидает увидеть ПЕРЕЧЕНЬ.
Если в контексте есть список кодов или конкретных пунктов, ТЫ ОБЯЗАН вывести этот список.
ИГНОРИРУЙ общие фразы из Уставов и Порядков, если есть фактический перечень.

ФОРМАТ ИСТОЧНИКОВ В КОНЦЕ ОТВЕТА:

В самом конце ответа ОБЯЗАТЕЛЬНО добавь раздел:

Использованные источники:
- [Полное название документа], [Раздел/Статья/Пункт]
- [Полное название документа], [Раздел/Статья/Пункт]

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:

{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:

{question}

ТВОЙ ОТВЕТ:"""

    return prompt


# ============================================================
# 7. СТРИМИНГ ОТ OLLAMA
# ============================================================

async def stream_ollama(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": 32768,
            "num_predict": 16384,
            "repeat_penalty": 1.05
        }
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
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
# 8. ЭНДПОИНТ
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = await build_prompt(request.question)

    async def generate_response():
        async for chunk in stream_ollama(prompt):
            yield chunk

    return StreamingResponse(
        generate_response(),
        media_type="text/plain; charset=utf-8"
    )


# ============================================================
# 9. ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Legal RAG API запущен и готов принимать запросы на http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )