"""Пример работы с памятью: extraction, storage, retrieval."""

import asyncio

from app.memory.memory_extractor import extract_memories
from app.memory.memory_service import MemoryService


async def main():
    service = MemoryService()
    user_id = "user_1"

    # 1. Извлечение memory из сообщений
    messages = [
        "Я использую Python для своего проекта.",
        "Мой проект — это курсовая по RAG.",
        "Я предпочитаю FastAPI вместо Flask.",
    ]

    for msg in messages:
        memories = extract_memories(msg)
        for category, text in memories:
            mid = service.add_memory(user_id=user_id, category=category, text=text)
            print(f"  Добавлена memory: [{category}] {text} (id={mid})")

    print()

    # 2. Retrieval
    query = "Какой фреймворк использовать?"
    results = service.retrieve_memory(query=query, user_id=user_id, top_k=3)
    print(f"Query: {query}")
    print(f"Найдено записей: {len(results)}\n")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['category']}: {r['text']}")

    print()

    # 3. Список всех записей
    all_memories = service.list_memories(user_id)
    print(f"Всего записей у {user_id}: {len(all_memories)}")
    for m in all_memories:
        print(f"  {m['id']}: [{m['category']}] {m['text']}")


if __name__ == "__main__":
    asyncio.run(main())
