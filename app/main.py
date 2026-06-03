"""FastAPI приложение RAG-ассистента."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path


if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, UploadFile, File
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
from app.config import settings
from app.ollama_client import OllamaClient, OllamaUnavailableError
from app.vector.qdrant_client import QdrantService
from app.vector.client import get_client
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import RerankerClient
from app.memory.memory_service import MemoryService
from app.memory.memory_store import MemoryStore
from app.memory.memory_extractor import extract_memories
from app.memory.conversation_memory import ConversationMemoryService
from app.rag.prompt_builder import build_prompt
from app.rag.document_loader import load_txt, load_pdf
from app.rag.chunker import chunk_text
from app.embeddings.embedder import generate_embeddings
from app.models import (
    AskRequest,
    AskResponse,
    MemoryCreate,
    MemoryItem,
    DocumentUploadResponse,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RetrievedChunkDebug,
    RetrievedMemory,
    MetricsResponse,
    HealthResponse,
    DeleteResponse,
    ClearConversationResponse,
)
from app.debug_endpoints import router as debug_router

ollama: OllamaClient | None = None
qdrant: QdrantService | None = None
retriever: HybridRetriever | None = None
reranker: RerankerClient | None = None
memory_service: MemoryService | None = None
conversation_service: ConversationMemoryService | None = None

logger = logging.getLogger("retrieval")
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler("logs/retrieval.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)




@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama, qdrant, retriever, reranker, memory_service, conversation_service
    print("\nRAG Memory Assistant\n")

    # 1. Ollama
    ollama = OllamaClient()
    ollama_ok = await ollama.health()
    print(f"Ollama:     {'OK' if ollama_ok else 'OFFLINE'} ({settings.ollama_model})")

    # 2. Qdrant embedded
    shared_client = get_client()
    print("Qdrant:     embedded")

    # 3. Embedding model
    try:
        from app.embeddings.embedder import get_dimension
        dim = get_dimension()
        print(f"Embeddings: OK (dim={dim})")
    except Exception as e:
        print(f"Embeddings: FAILED ({e})")

    # 4. Reranker
    reranker = RerankerClient()
    print(f"Reranker:   {'enabled' if reranker.enabled else 'disabled'}")

    # 5. Collections
    qdrant = QdrantService(client=shared_client)
    retriever = HybridRetriever(qdrant=qdrant)
    memory_store = MemoryStore(client=shared_client)
    memory_service = MemoryService(store=memory_store)
    conversation_service = ConversationMemoryService()

    print(f"Hybrid:     {'enabled' if settings.hybrid_search_enabled else 'disabled'}")
    print(f"\nAPI:        {'ready' if ollama_ok else 'ready (Ollama offline — /ask unavailable)'}")
    print(f"Docs:       http://127.0.0.1:8000/docs\n")
    yield


app = FastAPI(
    title="RAG Memory Assistant",
    description="Local FastAPI RAG assistant with hybrid search, long-term memory, and conversation context.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(debug_router)

@app.post("/clean/test/memories/{user_id}", tags=["debug"])
async def clear_user_memories(user_id: str):
    """Временная функция очистки всех памятей пользователя (для отладки)."""
    try:
        memory_service.store.client.delete(
            collection_name=memory_service.store.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
        )
        return {"deleted": True, "user_id": user_id, "message": "Все памяти пользователя удалены"}
    except Exception as e:
        return {"deleted": False, "error": str(e)}
    

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns reachability status for Ollama and Qdrant.",
    tags=["system"],
)
async def health():
    ollama_ok = await ollama.health()
    return HealthResponse(
        ollama=ollama_ok,
        ollama_host=ollama.host,
        ollama_model=settings.ollama_model,
        qdrant="embedded",
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Returns current configuration and indexed data counts.",
    tags=["system"],
)
async def metrics():
    doc_info = qdrant.client.count(collection_name=settings.qdrant_collection)
    mem_info = memory_service.store.client.count(collection_name=settings.memory_collection)
    return MetricsResponse(
        ollama_model=settings.ollama_model,
        embedding_model=settings.embedding_model,
        qdrant_mode="embedded",
        documents_collection=settings.qdrant_collection,
        memory_collection=settings.memory_collection,
        indexed_documents=0,
        indexed_chunks=doc_info.count,
        memory_entries=mem_info.count,
        retrieval_threshold=settings.retrieval_score_threshold,
        hybrid_enabled=settings.hybrid_search_enabled,
        reranker_enabled=settings.reranker_enabled,
        ollama_ctx=settings.ollama_num_ctx,
        embedding_batch_size=settings.embedding_batch_size,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    description="Main RAG endpoint. Retrieves relevant documents and memories, builds a prompt, and generates an answer via Ollama.",
    tags=["rag"],
)
async def ask(body: AskRequest):
    t_start = time.time()
    session_id = body.session_id
    
    # Динамически связываем сессию с пользователем
    user_id = session_id if session_id else "default_user"
    if user_id == "default":
        user_id = "test"
        session_id = "test"

    print(f"\n[RAG] Получен запрос от user_id: '{user_id}' (session: '{session_id}')", flush=True)

    # 1. Контекст диалога
    conv_messages, conv_summary = conversation_service.get_context(session_id)

    # 2. Поиск документов (База знаний RAG) с фильтрацией мусора
    t_retrieval = time.time()
    raw_documents = retriever.retrieve(body.question, top_k=body.top_k)
    if reranker.enabled:
        raw_documents = reranker.rerank(body.question, raw_documents)
        
    # Отсекаем чанки документов, у которых score ниже 0.4 (настраиваемый порог)
    # Это предотвратит попадание левых кусков из fastapi.txt в вопросы про хобби
    documents = [d for d in raw_documents if d.get("score", 0.0) >= 0.4]
    retrieval_time = time.time() - t_retrieval

    # 3. Поиск в долгосрочной памяти (Qdrant)
    memories = memory_service.retrieve_memory(
        query=body.question, user_id=user_id, top_k=3
    )

    print(f"[RAG] Найдено долгосрочных памятей для '{user_id}': {len(memories)}", flush=True)

    # 4. Сборка промпта — теперь memories передаются штатно, без хаков текста вопроса!
    system, prompt = build_prompt(
        question=body.question,
        documents=documents,
        memories=memories,
        conversation_messages=conv_messages,
        conversation_summary=conv_summary,
    )

    # 5. Генерация ответа через Ollama
    t_gen = time.time()
    try:
        answer = await ollama.generate(prompt=prompt, system=system)
    except OllamaUnavailableError as e:
        logger.error("Ollama unavailable: %s", e)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")
    generation_time = time.time() - t_gen

    # 6. Сохраняем ход беседы в историю
    conversation_service.add_turn(session_id, "user", body.question)
    conversation_service.add_turn(session_id, "assistant", answer)

    total_time = time.time() - t_start
    logger.info(
        "query=%r session=%s top_k=%d docs=%d memories=%d conv=%d "
        "prompt_size=%d retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f",
        body.question,
        session_id,
        body.top_k,
        len(documents),
        len(memories),
        len(conv_messages),
        len(prompt),
        retrieval_time * 1000,
        generation_time * 1000,
        total_time * 1000,
    )

    return AskResponse(
        answer=answer,
        sources=list({d["source_file"] for d in documents}),
        memories_used=[m["text"] for m in memories],
    )

@app.post(
    "/debug/retrieval",
    response_model=RetrievalDebugResponse,
    summary="Inspect retrieval pipeline",
    description="Returns raw retrieval results with scores, sources, and the final prompt size for debugging.",
    tags=["debug"],
)
async def debug_retrieval(body: RetrievalDebugRequest):
    user_id = body.session_id
    documents = retriever.retrieve(body.question, top_k=body.top_k)

    if reranker.enabled:
        docs_after_rerank = reranker.rerank(body.question, documents.copy())
    else:
        docs_after_rerank = documents

    system, prompt = build_prompt(
        question=body.question,
        documents=docs_after_rerank,
        memories=[],
    )

    return RetrievalDebugResponse(
        query=body.question,
        documents=[
            RetrievedChunkDebug(
                score=d["score"],
                text=d["text"],
                source_file=d["source_file"],
                chunk_index=d["chunk_index"],
                vector_score=d.get("vector_score", 0.0),
                bm25_score=d.get("bm25_score", 0.0),
                source=d.get("source", "vector"),
            )
            for d in docs_after_rerank
        ],
        memories=[],
        prompt_size=len(prompt),
        hybrid_enabled=settings.hybrid_search_enabled,
        reranker_enabled=settings.reranker_enabled,
    )


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    summary="Upload a document",
    description="Uploads a .txt or .pdf file, chunks it, generates embeddings, and indexes into Qdrant.",
    tags=["documents"],
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(400, "Only .txt and .pdf files are supported")

    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    save_path = docs_dir / file.filename

    content = await file.read()
    save_path.write_bytes(content)

    if file.filename.lower().endswith(".txt"):
        text = load_txt(save_path)
    else:
        text = load_pdf(save_path)

    chunks = list(
        chunk_text(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            source_file=file.filename,
        )
    )

    t_embed = time.time()
    embeddings = generate_embeddings([c["text"] for c in chunks], is_query=False)
    embed_time = time.time() - t_embed

    t_index = time.time()
    qdrant.index_documents(chunks, embeddings)
    index_time = time.time() - t_index

    total_time = embed_time + index_time
    logger.info(
        "upload=%s chunks=%d embed_ms=%.1f index_ms=%.1f total_ms=%.1f",
        file.filename,
        len(chunks),
        embed_time * 1000,
        index_time * 1000,
        total_time * 1000,
    )

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_count=len(chunks),
        indexing_time=round(total_time, 3),
    )


@app.post(
    "/memory",
    response_model=MemoryItem,
    summary="Create memory",
    description="Stores a long-term memory entry for a user.",
    tags=["memory"],
)
async def create_memory(body: MemoryCreate):
    memory_id = memory_service.add_memory(
        user_id=body.user_id,
        category=body.category,
        text=body.text,
    )
    return MemoryItem(
        id=memory_id,
        user_id=body.user_id,
        category=body.category,
        text=body.text,
    )


@app.get(
    "/memory",
    response_model=list[MemoryItem],
    summary="List memories",
    description="Returns all long-term memory entries for a user.",
    tags=["memory"],
)
async def list_memory(user_id: str = "default_user"):
    rows = memory_service.list_memories(user_id)
    return [
        MemoryItem(
            id=r["id"],
            user_id=r["user_id"],
            category=r["category"],
            text=r["text"],
        )
        for r in rows
    ]


@app.delete(
    "/memory/{memory_id}",
    response_model=DeleteResponse,
    summary="Delete memory",
    description="Deletes a memory entry by its ID.",
    tags=["memory"],
)
async def delete_memory(memory_id: str):
    ok = memory_service.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return DeleteResponse(deleted=True)


@app.post(
    "/conversation/{session_id}/clear",
    response_model=ClearConversationResponse,
    summary="Clear conversation",
    description="Clears the conversation history for a session.",
    tags=["conversation"],
)
async def clear_conversation(session_id: str):
    conversation_service.clear(session_id)
    return ClearConversationResponse(cleared=True)
