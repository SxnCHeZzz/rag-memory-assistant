"""Хранилище памяти пользователя в Qdrant (embedded/local mode)."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    PointStruct,
)

from app.config import settings
from app.embeddings.embedder import get_dimension
from app.vector.client import get_client



class MemoryStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str = settings.memory_collection,
    ):
        self.client = client or get_client()
        self.collection = collection
        self.vector_size = get_dimension()
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            # 1. Создаем коллекцию
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            # 2. Создаем индекс для user_id, чтобы фильтры в Embedded Qdrant не возвращали пустоту
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="user_id",
                field_schema=KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )

    def add(self, memory_id: str, vector: list[float], payload: dict):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
        )

    def get(self, memory_id: str) -> dict | None:
        try:
            result = self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
            )
            if result:
                point = result[0]
                return dict(point.payload) | {"id": str(point.id)}
            return None
        except Exception:
            return None

    def delete(self, memory_id: str) -> bool:
        try:
            from qdrant_client.models import PointIdsList
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=[memory_id]),
            )
            return True
        except Exception:
            return False

    def search(self, vector: list[float], user_id: str, top_k: int = 5) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
        )
        hits = [
            dict(hit.payload) | {"id": str(hit.id), "score": hit.score}
            for hit in results
        ]
        # === ВРЕМЕННЫЙ ДЕБАГ ===
        if hits:
            print(f"[Memory DEBUG] user_id={user_id} | query_vector_similarities: {[round(h['score'], 4) for h in hits]}")
        else:
            print(f"[Memory DEBUG] user_id={user_id} | NO memories found by filter")
        return hits

    def list_by_user(self, user_id: str, limit: int = 100) -> list[dict]:
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
            )
            
            # Подстраховка для Embedded Qdrant: 
            # Если вернулся кортеж (records, offset), берем records. Если список — оставляем как есть.
            if isinstance(scroll_result, tuple):
                points = scroll_result[0]
            else:
                points = scroll_result

            return [
                dict(point.payload) | {"id": str(point.id)}
                for point in points
            ]
