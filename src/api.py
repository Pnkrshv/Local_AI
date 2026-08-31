"""
FastAPI сервер для юридического RAG-ассистента.

Реализовано:
- гибридный поиск: векторный поиск + BM25;
- Reranker для точной пересортировки;
- обобщенный режим списков и подсчета пунктов;
- принудительная догрузка целых разделов;
- программный подсчет пунктов внутри релевантного раздела;
- разделение режимов ответа: количество / перечень;
- фильтрация некорректных и слишком коротких запросов.
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

from pathlib import Path
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
)

try:
    from config import RERANKER_MODEL_NAME
except ImportError:
    RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

from chunker import extract_list_stats, clean_legal_text


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

def is_valid_question(question: str) -> bool:
    """
    Проверяет, является ли запрос корректным и достаточно полным.
    Отсекает обрывки фраз, слишком короткие запросы и одиночные слова.
    """
    q = question.strip()
    if not q:
        return False
    
    words = q.split()
    
    # Меньше 3 слов - обычно не вопрос, а просто термин или обрывок
    if len(words) < 3:
        return False
        
    # Меньше 15 символов - слишком коротко для осмысленного юридического вопроса
    if len(q) < 15:
        return False
        
    return True


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


def get_meta_int(meta: dict, key: str, default: int = 0) -> int:
    try:
        value = (meta or {}).get(key, default)

        if value is None:
            return default

        return int(value)
    except Exception:
        return default


def merge_docs_dedup(docs: list[str]) -> str:
    lines = []
    seen = set()

    for doc in docs:
        if not doc:
            continue

        for line in doc.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            normalized = re.sub(r"\s+", " ", stripped)

            if normalized in seen:
                continue

            seen.add(normalized)
            lines.append(stripped)

    return "\n".join(lines)


def section_prefix(section: str) -> str:
    if not section:
        return ""

    parts = [p.strip() for p in section.split("/")]

    if not parts:
        return section

    last = parts[-1]

    m = re.match(
        r"((?:Пункт|Статья|Глава|Раздел|Приложение)\s*[0-9IVXLC]+)",
        last,
        re.IGNORECASE
    )

    if m:
        parts[-1] = m.group(1)
        return " / ".join(parts)

    if len(parts) == 1:
        m = re.match(
            r"([А-Яа-яA-Za-z]+\s+[0-9IVXLC]+)",
            parts[0],
            re.IGNORECASE
        )

        if m:
            return m.group(1)

    return section


def fetch_section_sync(source: str, section: str) -> list[str]:
    try:
        res = collection.get(
            where={
                "source": source,
                "section": section,
            }
        )

        docs = res.get("documents") or []
        ids = res.get("ids") or []

        if docs and len(docs) >= 2:
            items = sorted(zip(ids, docs), key=lambda x: int(x[0].split("_")[1]) if x[0].startswith("chunk_") else 0)
            return [doc for _, doc in items]

    except Exception:
        pass

    try:
        all_res = collection.get(where={"source": source})

        all_ids = all_res.get("ids") or []
        all_docs = all_res.get("documents") or []
        all_metas = all_res.get("metadatas") or []

        prefix = section_prefix(section)

        filtered = []

        for id_, doc, meta in zip(all_ids, all_docs, all_metas):
            sec = (meta or {}).get("section", "")

            if sec and sec.startswith(prefix):
                filtered.append((id_, doc))

        if filtered:
            filtered.sort(key=lambda x: int(x[0].split("_")[1]) if x[0].startswith("chunk_") else 0)
            return [doc for _, doc in filtered]

    except Exception:
        pass

    return []


# ============================================================
# 5. ГИБРИДНЫЙ ПОИСК
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

    v_ids = vector_res.get("ids", [[]])[0] if vector_res.get("ids") else []
    v_docs = vector_res.get("documents", [[]])[0] if vector_res.get("documents") else []
    v_metas = vector_res.get("metadatas", [[]])[0] if vector_res.get("metadatas") else []

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

    ranked_ids = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return (
        [docs_by_id[id_] for id_ in ranked_ids],
        [metas_by_id[id_] for id_ in ranked_ids]
    )


# ============================================================
# 6. ИНТЕНТЫ
# ============================================================

def is_count_intent(question: str) -> bool:
    q = question.lower()

    return bool(
        re.search(
            r"\b(сколько|количество|кол-во|подсчитай|посчитай|число|пересчитай)\b",
            q
        )
    )


def is_list_intent(question: str) -> bool:
    q = question.lower()

    return bool(
        re.search(
            r"\b("
            r"какие|какой|какая|какое|"
            r"перечисли|перечень|список|"
            r"полномочия|права|обязанности|"
            r"специальности|направления|стандарты|"
            r"функции|задачи"
            r")\b",
            q
        )
    )


def get_response_mode(question: str) -> str:
    count_intent = is_count_intent(question)
    list_intent = is_list_intent(question)

    if count_intent and not list_intent:
        return "count_only"

    if list_intent and not count_intent:
        return "list_items"

    if count_intent and list_intent:
        return "list_items"

    return "auto"


# ============================================================
# 7. ЛОГИКА ПОИСКА И ПРОМПТА
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

    response_mode = get_response_mode(question)

    count_intent = response_mode == "count_only"
    list_intent = response_mode == "list_items"

    expand_intent = count_intent or list_intent

    first_meta_by_key = {}
    first_doc_by_key = {}
    unique_sections = []

    for doc, meta in top_pairs:
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        key = (source, section)

        if key not in first_meta_by_key:
            first_meta_by_key[key] = meta
            first_doc_by_key[key] = doc

        if section != "Без раздела" and key not in unique_sections:
            unique_sections.append(key)

    if expand_intent:
        candidate_keys = unique_sections[:6]
    else:
        section_counts = defaultdict(int)

        for _, meta in top_pairs:
            source = meta.get("source", "")
            section = meta.get("section", "Без раздела")

            if section == "Без раздела":
                continue

            section_counts[(source, section)] += 1

        candidate_keys = [
            key for key, count in section_counts.items()
            if count >= 2
        ]

    section_infos = []

    for key in candidate_keys:
        source, section = key

        docs = await run_in_thread(fetch_section_sync, source, section)

        if not docs:
            docs = [first_doc_by_key.get(key, "")]

        merged_text = merge_docs_dedup(docs)
        full_text = clean_legal_text(merged_text, source)

        meta = first_meta_by_key.get(key, {})

        list_count = get_meta_int(meta, "list_count", 0)
        primary_count = get_meta_int(meta, "primary_count", 0)
        supplemental_count = get_meta_int(meta, "supplemental_count", 0)

        if list_count <= 0:
            stats = extract_list_stats(full_text, [])

            list_count = stats["list_count"]
            primary_count = stats["primary_count"]
            supplemental_count = stats["supplemental_count"]

        if primary_count <= 0 and list_count > 0:
            primary_count = list_count

        section_infos.append(
            {
                "source": source,
                "section": section,
                "text": full_text,
                "list_count": list_count,
                "primary_count": primary_count,
                "supplemental_count": supplemental_count,
            }
        )

    primary_info = None

    if section_infos:
        valid_infos = [info for info in section_infos if info["primary_count"] > 0]

        if valid_infos:
            primary_info = max(
                valid_infos,
                key=lambda x: x["primary_count"]
            )

    context_pairs = []
    seen_context = set()

    if primary_info and (expand_intent or primary_info.get("primary_count", 0) > 0):
        key = (primary_info["source"], primary_info["section"])

        context_pairs.append(
            (
                primary_info["text"],
                {
                    "source": primary_info["source"],
                    "section": primary_info["section"],
                }
            )
        )

        seen_context.add(key)

    for info in sorted(
        section_infos,
        key=lambda x: x.get("primary_count", 0),
        reverse=True
    ):
        key = (info["source"], info["section"])

        if key in seen_context:
            continue

        context_pairs.append(
            (
                info["text"],
                {
                    "source": info["source"],
                    "section": info["section"],
                }
            )
        )

        seen_context.add(key)

    for doc, meta in top_pairs:
        source = meta.get("source", "")
        section = meta.get("section", "Без раздела")
        key = (source, section)

        if key in seen_context:
            continue

        context_pairs.append((doc, meta))
        seen_context.add(key)

    context_parts = []
    total_len = 0

    for i, (doc, meta) in enumerate(context_pairs):
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

    hint_lines = []

    if primary_info:
        hint_lines.append("СЛУЖЕБНАЯ ИНФОРМАЦИЯ ОТ АЛГОРИТМА АНАЛИЗА ДОКУМЕНТОВ:")
        hint_lines.append(
            f"Наиболее релевантный раздел: {primary_info['source']} / {primary_info['section']}"
        )

        if primary_info.get("primary_count", 0) > 0:
            hint_lines.append(
                f"Основных пунктов в этом разделе: {primary_info['primary_count']}."
            )

        if primary_info.get("supplemental_count", 0) > 0:
            hint_lines.append(
                f"Дополнительных подпунктов в этом разделе: {primary_info['supplemental_count']}."
            )

        if response_mode == "count_only":
            hint_lines.append(
                "Пользователь спросил КОЛИЧЕСТВО. Ответь только количеством, не перечисляй пункты."
            )
        elif response_mode == "list_items":
            hint_lines.append(
                "Пользователь спросил ПЕРЕЧЕНЬ. Обязательно перечисли пункты списка. "
                "Если пунктов больше 10 — выведи первые 10 и добавь примечание, что список неполный."
            )

    hints = "\n".join(hint_lines)

    prompt = f"""Ты — строгий и внимательный юридический ассистент. Твоя задача — отвечать на вопросы пользователя ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов нормативно-правовых актов и служебной информации от алгоритма анализа документов.

{hints}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. Никаких внутренних знаний: Используй ТОЛЬКО информацию из раздела "Контекст" и служебной информации от алгоритма. Если ответа нет, напиши: "В предоставленных документах нет информации для ответа на этот вопрос."

2. Режим ответа:
- Если пользователь спрашивает «сколько», «количество» — ответь только количеством.
- Если пользователь спрашивает «какие», «перечисли», «что входит» — перечисли пункты списка.
- Запрещено отвечать только количеством, если пользователь просит перечислить.

3. Использование алгоритмического подсчета:
- Если в служебной информации указано число пунктов, используй именно его для вопросов о количестве.
- Запрещено писать другое число.
- Запрещено путать номер пункта с количеством пунктов.

4. Правило вывода списков:
- Если пунктов больше 10 — выведи РОВНО первые 10 пунктов.
- После 10-го пункта напиши: "Это не полный список. Полный перечень содержится в [название документа], [раздел/пункт]."
- Если пунктов 10 или меньше — выведи все.

5. Цитирование источников:
- Все источники указывай ТОЛЬКО один раз в самом конце ответа в разделе "Использованные источники".

6. Запрет на выдумки:
- Никогда не придумывай номера статей, пунктов, дат или фактов.

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

    try:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        debug_path = debug_dir / "last_prompt.txt"
        debug_path.write_text(prompt, encoding="utf-8")

        print(
            f"DEBUG: response_mode={response_mode}, "
            f"primary_section={primary_info['section'] if primary_info else None}, "
            f"primary_count={primary_info.get('primary_count', 0) if primary_info else 0}, "
            f"saved to {debug_path}"
        )
    except Exception:
        pass

    return prompt


# ============================================================
# 8. СТРИМИНГ ОТ OLLAMA
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
# 9. ЭНДПОИНТ
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question.strip()
    
    # ПРОВЕРКА НА ВАЛИДНОСТЬ ЗАПРОСА
    if not is_valid_question(question):
        async def invalid_response():
            yield "Ваш запрос сформулирован некорректно, слишком коротко или содержит опечатки. Пожалуйста, задайте полный и четкий вопрос по нормативно-правовым актам."
        return StreamingResponse(invalid_response(), media_type="text/plain; charset=utf-8")

    prompt = await build_prompt(question)

    async def generate_response():
        async for chunk in stream_ollama(prompt):
            yield chunk

    return StreamingResponse(
        generate_response(),
        media_type="text/plain; charset=utf-8"
    )


# ============================================================
# 10. ТОЧКА ВХОДА
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