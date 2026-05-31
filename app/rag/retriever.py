"""Ретривер для поиска релевантных чанков с score threshold."""

from app.config import settings
from app.embeddings.embedder import generate_embeddings
from app.vector.qdrant_client import QdrantService


class Retriever:
    def __init__(self, qdrant: QdrantService | None = None):
        self.qdrant = qdrant or QdrantService()
        self.threshold = settings.retrieval_score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Находит top-k релевантных чанков с фильтрацией по threshold."""
        embeddings = generate_embeddings([query], is_query=True)
        results = self.qdrant.search(vector=embeddings[0], top_k=top_k)
        filtered = [
            {
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "source_file": hit.payload.get("source_file", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
            }
            for hit in results
            if hit.score >= self.threshold
        ]
        return filtered
