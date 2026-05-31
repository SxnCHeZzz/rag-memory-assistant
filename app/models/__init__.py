"""Pydantic models for API request/response validation and OpenAPI docs."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""

    question: str = Field(..., description="User question in natural language", examples=["Какие преимущества FastAPI?"])
    top_k: int = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")
    session_id: str = Field("default", description="Conversation session identifier")


class AskResponse(BaseModel):
    """Response from /ask endpoint."""

    answer: str = Field(..., description="Generated answer based on retrieved context")
    sources: list[str] = Field(default_factory=list, description="List of source filenames used for the answer")
    memories_used: list[str] = Field(default_factory=list, description="Long-term memory entries that influenced the answer")


class MemoryCreate(BaseModel):
    """Request body for creating a memory entry."""

    user_id: str = Field(..., description="User identifier")
    category: str = Field(..., description="Memory category (e.g. preference, fact, context)", examples=["preference"])
    text: str = Field(..., description="Memory content", examples=["Я предпочитаю краткие ответы"])


class MemoryItem(BaseModel):
    """Single memory entry response."""

    id: str = Field(..., description="Unique memory UUID")
    user_id: str = Field(..., description="Owner user ID")
    category: str = Field(..., description="Memory category")
    text: str = Field(..., description="Memory content")
    score: float | None = Field(None, description="Similarity score when returned from search")


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    filename: str = Field(..., description="Uploaded file name")
    chunks_count: int = Field(..., description="Number of text chunks extracted")
    indexing_time: float = Field(..., description="Total embedding + index time in seconds")


class RetrievedChunkDebug(BaseModel):
    """Retrieved document chunk with debug scoring information."""

    score: float = Field(..., description="Final fused score (RRF)")
    text: str = Field(..., description="Chunk text content")
    source_file: str = Field(..., description="Source document filename")
    chunk_index: int = Field(..., description="Chunk index within the document")
    vector_score: float = Field(0.0, description="Raw vector similarity score")
    bm25_score: float = Field(0.0, description="Raw BM25 keyword score")
    source: str = Field("vector", description="Retrieval source: vector, bm25, hybrid, reranked")


class RetrievedMemory(BaseModel):
    """Retrieved memory entry with similarity score."""

    score: float = Field(..., description="Similarity score")
    id: str = Field(..., description="Memory UUID")
    text: str = Field(..., description="Memory content")
    category: str = Field(..., description="Memory category")


class RetrievalDebugRequest(BaseModel):
    """Request body for /debug/retrieval endpoint."""

    question: str = Field(..., description="Query to test retrieval", examples=["Какие преимущества FastAPI?"])
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")


class RetrievalDebugResponse(BaseModel):
    """Response from /debug/retrieval with full pipeline inspection."""

    query: str = Field(..., description="Original query")
    documents: list[RetrievedChunkDebug] = Field(default_factory=list, description="Retrieved document chunks")
    memories: list[RetrievedMemory] = Field(default_factory=list, description="Retrieved memory entries")
    prompt_size: int = Field(..., description="Final prompt length in characters")
    hybrid_enabled: bool = Field(..., description="Whether hybrid search is active")
    reranker_enabled: bool = Field(..., description="Whether reranker is active")


class MetricsResponse(BaseModel):
    """System metrics and configuration snapshot."""

    ollama_model: str = Field(..., description="Active LLM model name")
    embedding_model: str = Field(..., description="Active embedding model name")
    qdrant_mode: str = Field(..., description="Qdrant mode: embedded or server")
    documents_collection: str = Field(..., description="Qdrant collection for documents")
    memory_collection: str = Field(..., description="Qdrant collection for memories")
    indexed_documents: int = Field(..., description="Count of uploaded documents")
    indexed_chunks: int = Field(..., description="Count of indexed chunks")
    memory_entries: int = Field(..., description="Count of stored memory entries")
    retrieval_threshold: float = Field(..., description="Score threshold for retrieval filtering")
    hybrid_enabled: bool = Field(..., description="Hybrid search enabled")
    reranker_enabled: bool = Field(..., description="Reranker enabled")
    ollama_ctx: int = Field(..., description="LLM context window size")
    embedding_batch_size: int = Field(..., description="Batch size for embedding generation")


class HealthResponse(BaseModel):
    """Health check response."""

    ollama: bool = Field(..., description="Whether Ollama is reachable")
    ollama_host: str = Field(..., description="Resolved Ollama host URL")
    ollama_model: str = Field(..., description="Configured model name")
    qdrant: str = Field(..., description="Qdrant status")


class DeleteResponse(BaseModel):
    """Generic deletion confirmation."""

    deleted: bool = Field(..., description="Whether the item was successfully deleted")


class ClearConversationResponse(BaseModel):
    """Conversation clear confirmation."""

    cleared: bool = Field(..., description="Whether the conversation was successfully cleared")


# ---------------------------------------------------------------------------
# Debug-specific response models
# ---------------------------------------------------------------------------


class EncodingInfoResponse(BaseModel):
    """System encoding diagnostic information."""

    python_version: str
    sys_stdout_encoding: str
    sys_stderr_encoding: str
    platform: str
    test_russian: str
    test_unicode: str
    message: str


class EncodingBytesResponse(BaseModel):
    """UTF-8 byte-level encoding verification payload."""

    text: str
    utf8_bytes_hex: str
    utf8_bytes_b64: str
    expected_hex: str


class QdrantRawResponse(BaseModel):
    """Raw Qdrant payload inspection response."""

    payload_text: str | None = None
    payload_text_hex: str | None = None
    payload_text_b64: str | None = None
    source_file: str | None = None
    is_ascii_only: bool | None = None
    error: str | None = None


class ProjectContextResponse(BaseModel):
    """Structured project metadata for documentation generation."""

    project_overview: dict
    architecture: dict
    technologies: dict
    rag_pipeline: list[str]
    memory_system: dict
    retrieval_metrics: dict
    api_endpoints: dict[str, str]
    dependencies: list[str]
    future_improvements: list[str]


# ---------------------------------------------------------------------------
# Debug-specific response models
# ---------------------------------------------------------------------------


class EncodingInfoResponse(BaseModel):
    """System encoding diagnostic information."""

    python_version: str
    sys_stdout_encoding: str
    sys_stderr_encoding: str
    platform: str
    test_russian: str
    test_unicode: str
    message: str


class EncodingBytesResponse(BaseModel):
    """UTF-8 byte-level encoding verification payload."""

    text: str
    utf8_bytes_hex: str
    utf8_bytes_b64: str
    expected_hex: str


class QdrantRawResponse(BaseModel):
    """Raw Qdrant payload inspection response."""

    payload_text: str | None = None
    payload_text_hex: str | None = None
    payload_text_b64: str | None = None
    source_file: str | None = None
    is_ascii_only: bool | None = None
    error: str | None = None


class ProjectContextResponse(BaseModel):
    """Structured project metadata for documentation generation."""

    project_overview: dict
    architecture: dict
    technologies: dict
    rag_pipeline: list[str]
    memory_system: dict
    retrieval_metrics: dict
    api_endpoints: dict[str, str]
    dependencies: list[str]
    future_improvements: list[str]
