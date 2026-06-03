"""Сервис для управления памятью пользователя."""

import uuid
from datetime import datetime, timezone

from app.memory.memory_store import MemoryStore
from app.embeddings.embedder import generate_embeddings


class MemoryService:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore()

    def _make_id(self, user_id: str, category: str, text: str) -> str:
        """Генерирует детерминированный UUID на основе текста, чтобы избежать дубликатов."""
        # Очистим текст от лишних пробелов и приведем к нижнему регистру для стабильности хэша
        normalized_text = " ".join(text.lower().split())
        key_string = f"{user_id}_{category}_{normalized_text}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key_string))

    def add_memory(self, user_id: str, category: str, text: str) -> str:
        """Добавляет запись в память пользователя."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        memory_id = self._make_id(user_id, category, text)

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
        """Железобетонный поиск памяти для Embedded Qdrant."""
        # Напрямую запрашиваем список всех памятей пользователя через scroll
        try:
            all_memories = self.store.list_by_user(user_id, limit=100)
        except Exception:
            all_memories = []

        # Если у пользователя всего одна-две записи, не мучаем векторный поиск — отдаем их
        if all_memories and len(all_memories) <= 3:
            return all_memories[:top_k]

        # Если записей много, делаем простейший текстовый поиск по совпадению слов
        query_words = set(query.lower().replace("?", "").split())
        matched = []
        for m in all_memories:
            text_lower = m.get("text", "").lower()
            if any(word in text_lower for word in query_words if len(word) > 2):
                matched.append(m)

        if matched:
            return matched[:top_k]

        # В крайнем случае используем векторный поиск
        embeddings = generate_embeddings([query], is_query=True)
        return self.store.search(vector=embeddings[0], user_id=user_id, top_k=top_k)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Удаляет запись памяти."""
        return self.store.delete(memory_id)

    def list_memories(self, user_id: str, limit: int = 100) -> list[dict]:
        """Возвращает все записи памяти пользователя."""
        return self.store.list_by_user(user_id, limit=limit)
