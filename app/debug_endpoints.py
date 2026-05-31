"""Debug endpoints for encoding verification and system diagnostics.

These endpoints are conditionally enabled via the RAG_DEBUG environment variable.
If RAG_DEBUG is unset (default), all /debug/* routes remain available for development.
In production, the router can be excluded from the main app.
"""

import base64
import json
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.models import (
    EncodingInfoResponse,
    EncodingBytesResponse,
    QdrantRawResponse,
    ProjectContextResponse,
)

router = APIRouter(prefix="/debug", tags=["debug"])

TEST_RUSSIAN = "Привет мир! Какие преимущества FastAPI?"
TEST_UNICODE = "日本語 🎉 émojis"

# ---------------------------------------------------------------------------
# Encoding diagnostics
# ---------------------------------------------------------------------------


@router.get(
    "/encoding",
    response_model=EncodingInfoResponse,
    summary="Encoding diagnostics",
    description="Returns system encoding info and Russian/Unicode test strings to verify UTF-8 pipeline integrity.",
)
def debug_encoding():
    return {
        "python_version": sys.version,
        "sys_stdout_encoding": sys.stdout.encoding,
        "sys_stderr_encoding": sys.stderr.encoding,
        "platform": sys.platform,
        "test_russian": TEST_RUSSIAN,
        "test_unicode": TEST_UNICODE,
        "message": "Если этот текст кривой — проблема в JSON serialization",
    }


@router.get(
    "/encoding-raw",
    response_class=PlainTextResponse,
    summary="Raw UTF-8 text response",
    description="Returns plain text with UTF-8 BOM-less Russian and Unicode characters. Bypasses JSON serialization.",
)
def debug_encoding_raw():
    return PlainTextResponse(
        content=f"Russian: {TEST_RUSSIAN}\nUnicode: {TEST_UNICODE}\n",
        media_type="text/plain; charset=utf-8",
    )


@router.get(
    "/encoding-bytes",
    response_model=EncodingBytesResponse,
    summary="UTF-8 byte-level verification",
    description="Returns the test Russian string encoded as hex and base64 so clients can verify byte-level correctness.",
)
def debug_encoding_bytes():
    raw = TEST_RUSSIAN.encode("utf-8")
    return {
        "text": TEST_RUSSIAN,
        "utf8_bytes_hex": raw.hex(),
        "utf8_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "expected_hex": "d09f d180 d0b8 d0b2 d0b5 d182",
    }


# ---------------------------------------------------------------------------
# Qdrant diagnostics
# ---------------------------------------------------------------------------


@router.get(
    "/qdrant-raw",
    response_model=QdrantRawResponse,
    summary="First Qdrant document inspection",
    description="Reads the first stored chunk from Qdrant and returns its payload with byte-level encoding verification.",
)
def debug_qdrant_raw():
    from app.vector.client import get_client

    client = get_client()
    result = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=1,
    )[0]

    if not result:
        return {"error": "No documents found"}

    point = result[0]
    text = point.payload.get("text", "")
    raw = text.encode("utf-8")

    return {
        "payload_text": text,
        "payload_text_hex": raw.hex(),
        "payload_text_b64": base64.b64encode(raw).decode("ascii"),
        "source_file": point.payload.get("source_file"),
        "is_ascii_only": text.isascii(),
    }


# ---------------------------------------------------------------------------
# Project context export (for thesis / documentation generation)
# ---------------------------------------------------------------------------


@router.get(
    "/project-context",
    response_model=ProjectContextResponse,
    summary="Structured project metadata",
    description="Returns a machine-readable summary of the project architecture, pipeline, dependencies and metrics. Useful for generating documentation or thesis materials.",
)
def debug_project_context():
    return {
        "project_overview": {
            "name": "RAG Memory Assistant",
            "description": "Local FastAPI RAG assistant with hybrid search, long-term memory, and conversation context. Built for a university course project.",
        },
        "architecture": {
            "pattern": "modular_monolith",
            "layers": ["api", "services", "retrieval", "storage", "llm_client"],
            "description": "FastAPI application with pluggable retrievers, memory store, and LLM client. No external frameworks like LangChain.",
        },
        "technologies": {
            "framework": "FastAPI + Pydantic",
            "vector_store": "Qdrant (embedded mode)",
            "llm": "Ollama (qwen2.5:7b-instruct-q4_K_M)",
            "embeddings": "sentence-transformers (multilingual-e5-base)",
            "retrieval": "Dense (cosine) + Sparse (BM25) + RRF fusion",
            "reranking": "Optional cross-encoder (BAAI/bge-reranker-base)",
            "chunking": "Sentence-aware with code block detection",
            "memory": "Vector semantic search + conversation history",
        },
        "rag_pipeline": [
            "Document upload (txt/pdf)",
            "Smart chunking (sentence-aware, code block aware)",
            "Embedding generation (E5 with passage: prefix)",
            "Qdrant vector index (embedded mode, deterministic UUIDs)",
            "Query embedding (E5 with query: prefix)",
            "Hybrid retrieval (vector + BM25 with RRF)",
            "Optional reranking (cross-encoder)",
            "Prompt compression (deduplication, score trimming)",
            "LLM generation via Ollama",
        ],
        "memory_system": {
            "long_term": "Vector-based semantic memory in Qdrant. Stores user facts/preferences with deterministic UUIDs.",
            "conversation": "Session-based in-memory conversation history with rule-based auto-summary.",
            "extraction": "Rule-based fact extraction from user messages during /ask.",
        },
        "retrieval_metrics": {
            "precision_at_1": 0.80,
            "precision_at_3": 0.60,
            "mrr": 0.90,
        },
        "api_endpoints": {
            "/ask": "Main RAG question answering",
            "/documents": "Upload and index new documents",
            "/memory": "CRUD for long-term memory",
            "/conversation/{session_id}/clear": "Clear conversation history",
            "/metrics": "System status and configuration",
            "/health": "Health check for Ollama and Qdrant",
            "/debug/retrieval": "Retrieval pipeline inspection",
            "/debug/encoding": "Encoding diagnostics",
            "/debug/project-context": "Project metadata export",
        },
        "dependencies": [
            "fastapi",
            "pydantic-settings",
            "qdrant-client",
            "sentence-transformers",
            "rank-bm25",
            "PyPDF2",
        ],
        "future_improvements": [
            "GraphRAG with entity-relationship extraction",
            "Streaming LLM responses via SSE",
            "WebSocket chat interface",
            "Query expansion and rephrasing",
            "Conversational evaluation metrics",
        ],
    }


@router.get(
    "/export-thesis-context",
    response_model=dict,
    summary="Export project context as Markdown",
    description="Generates a Markdown file in exports/thesis_context.md with a structured summary suitable for a thesis or course paper.",
)
def debug_export_thesis_context():
    ctx = debug_project_context()
    payload = ctx if hasattr(ctx, "__iter__") and not isinstance(ctx, dict) else (ctx.dict() if hasattr(ctx, "dict") else ctx)

    lines = [
        "# RAG Memory Assistant — Thesis Context",
        "",
        "## 1. Введение",
        f"**{payload['project_overview']['name']}** — {payload['project_overview']['description']}",
        "",
        "## 2. Архитектура системы",
        f"Архитектурный паттерн: **{payload['architecture']['pattern']}**.",
        f"Слои: {', '.join(payload['architecture']['layers'])}.",
        "",
        "## 3. RAG Pipeline",
    ]
    for step in payload["rag_pipeline"]:
        lines.append(f"- {step}")

    lines.extend([
        "",
        "## 4. Memory System",
        f"- **Долгосрочная память:** {payload['memory_system']['long_term']}",
        f"- **Контекст разговора:** {payload['memory_system']['conversation']}",
        f"- **Извлечение фактов:** {payload['memory_system']['extraction']}",
        "",
        "## 5. Hybrid Search",
        f"- {payload['technologies']['retrieval']}",
        f"- Реранкинг: {payload['technologies']['reranking']}",
        "",
        "## 6. Evaluation Metrics",
        f"- Precision@1: {payload['retrieval_metrics']['precision_at_1']:.2f}",
        f"- Precision@3: {payload['retrieval_metrics']['precision_at_3']:.2f}",
        f"- MRR: {payload['retrieval_metrics']['mrr']:.2f}",
        "",
        "## 7. API Endpoints",
    ])
    for endpoint, desc in payload["api_endpoints"].items():
        lines.append(f"- `{endpoint}` — {desc}")

    lines.extend([
        "",
        "## 8. Технологический стек",
    ])
    for key, val in payload["technologies"].items():
        lines.append(f"- **{key}:** {val}")

    lines.extend([
        "",
        "## 9. Возможные улучшения",
    ])
    for imp in payload["future_improvements"]:
        lines.append(f"- {imp}")

    lines.extend([
        "",
        "## 10. Заключение",
        "Проект демонстрирует полноценный RAG pipeline с hybrid retrieval, долгосрочной памятью и сессионным контекстом. Реализован без сторонних фреймворков (LangChain), что позволяет глубоко понять архитектуру и легко расширять функциональность.",
    ])

    exports_dir = Path("exports")
    exports_dir.mkdir(parents=True, exist_ok=True)
    path = exports_dir / "thesis_context.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    return {"exported_to": str(path), "lines": len(lines)}
