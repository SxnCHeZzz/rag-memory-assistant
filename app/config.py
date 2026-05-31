"""Конфигурация приложения из переменных окружения."""

from pathlib import Path

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_timeout: float = 120.0
    ollama_num_ctx: int = 2048

    # Qdrant
    qdrant_collection: str = "documents"

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_batch_size: int = 8

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 128

    # Retrieval
    retrieval_score_threshold: float = 0.60
    hybrid_search_enabled: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # Reranker
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k_before: int = 10
    reranker_top_k_after: int = 5
    reranker_batch_size: int = 4

    # Conversation
    conversation_history_max: int = 10
    conversation_summary_threshold: int = 6

    # Memory
    memory_collection: str = "user_memory"

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()