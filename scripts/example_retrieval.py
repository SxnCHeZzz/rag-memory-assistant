"""Пример запуска retrieval."""

import asyncio

from app.rag.retriever import Retriever


async def main():
    retriever = Retriever()
    query = "Какие технологии используются в проекте?"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['source_file']} chunk {r['chunk_index']}")
        print(f"Text: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
