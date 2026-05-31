"""Сервис для управления памятью пользователя."""

import uuid
from datetime import datetime, timezone

from app.memory.memory_store import MemoryStore
from app.embeddings.embedder import generate_embeddings


class MemoryService:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore()

    def _make_id(self, user_id: str, category: str, timestamp: str) -> str:
        """Генерирует детерминированный UUID из строки (embedded mode требует UUID)."""
        text = f"{user_id}_{category}_{timestamp}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def add_memory(self, user_id: str, category: str, text: str) -> str:
        """Добавляет запись в память пользователя."""
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_id = self._make_id(user_id, category, timestamp)

        embeddings = generate_embeddings([text], is_query=False)

        self.store.add(
            memory_id=memory_id,
            vector=embeddings[0],
            payload={
                "user_id": user_id,
                "category": category,
                "text": text,
                "timestamp": timestamp,
            },
        )
        return memory_id

    def retrieve_memory(self, query: str, user_id: str, top_k: int = 3) -> list[dict]:
        """Находит релевантные записи памяти пользователя."""
        embeddings = generate_embeddings([query], is_query=True)
        return self.store.search(vector=embeddings[0], user_id=user_id, top_k=top_k)

    def delete_memory(self, memory_id: str) -> bool:
        """Удаляет запись памяти."""
        return self.store.delete(memory_id)

    def list_memories(self, user_id: str, limit: int = 100) -> list[dict]:
        """Возвращает все записи памяти пользователя."""
        return self.store.list_by_user(user_id, limit=limit)
