"""Скрипт индексации документов в Qdrant."""

import time
import logging
from pathlib import Path

from app.config import settings
from app.rag.document_loader import iter_documents
from app.rag.chunker import chunk_text
from app.embeddings.embedder import generate_embeddings
from app.vector.qdrant_client import QdrantService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    docs_dir = Path("data/documents")
    if not docs_dir.exists():
        logger.warning("Директория %s не найдена, создаём...", docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Положите .txt или .pdf файлы в %s и запустите снова.", docs_dir)
        return

    qdrant = QdrantService()

    docs = list(iter_documents(docs_dir))
    logger.info("Загружено документов: %d", len(docs))

    all_chunks = []
    for doc in docs:
        chunks = list(
            chunk_text(
                doc["text"],
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                source_file=doc["source_file"],
            )
        )
        all_chunks.extend(chunks)

    logger.info("Всего чанков: %d", len(all_chunks))

    if not all_chunks:
        logger.info("Нет чанков для индексации.")
        return

    start = time.time()
    embeddings = generate_embeddings([c["text"] for c in all_chunks], is_query=False)
    embedding_time = time.time() - start
    logger.info("Embeddings сгенерированы (dim=%d) за %.2f сек", len(embeddings[0]), embedding_time)

    start = time.time()
    count = qdrant.index_documents(all_chunks, embeddings)
    index_time = time.time() - start
    logger.info("Загружено %d чанков в Qdrant за %.2f сек", count, index_time)

    total_time = embedding_time + index_time
    logger.info("Общее время индексации: %.2f сек", total_time)


if __name__ == "__main__":
    main()
