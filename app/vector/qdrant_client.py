"""Клиент для Qdrant с поддержкой batch indexing (embedded/local mode)."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.embeddings.embedder import get_dimension
from app.vector.client import get_client


class QdrantService:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str = settings.qdrant_collection,
    ):
        self.client = client or get_client()
        self.collection = collection
        self.vector_size = get_dimension()
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _make_uuid(self, text: str) -> str:
        """Генерирует детерминированный UUID из строки (embedded mode требует UUID)."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def upsert(self, id: str | int, vector: list[float], payload: dict | None = None):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=id, vector=vector, payload=payload or {})],
        )

    def index_documents(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """Batch upsert чанков с embeddings."""
        points = [
            PointStruct(
                id=self._make_uuid(f"{chunk['source_file']}_{chunk['chunk_index']}"),
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "character_count": chunk["character_count"],
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, vector: list[float], top_k: int = 5):
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
        )
