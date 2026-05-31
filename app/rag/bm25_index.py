"""Lightweight BM25 index на CPU, без внешних зависимостей."""

import json
import math
import re
from pathlib import Path
from typing import Iterator

from app.config import settings

_DATA_DIR = Path("./bm25_data")
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> list[str]:
    """Простая токенизация: lowercase, alphanumeric only."""
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


class BM25Index:
    """In-memory BM25. Перестраивается при каждом поиске — для малого датасета нормально."""

    def __init__(self):
        self.k1 = 1.5
        self.b = 0.75
        self._chunks: list[dict] = []
        self._build_index()

    def _build_index(self):
        """Собирает чанки из Qdrant payload."""
        from app.vector.client import get_client

        client = get_client()
        collection = settings.qdrant_collection
        if not client.collection_exists(collection):
            return

        all_ids = []
        offset = None
        while True:
            result, offset = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
            )
            for point in result:
                all_ids.append(point.id)
            if offset is None:
                break

        if not all_ids:
            return

        points = client.retrieve(collection_name=collection, ids=all_ids)
        self._chunks = []
        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            self._chunks.append({
                "id": str(p.id),
                "text": text,
                "source_file": payload.get("source_file", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "tokens": _tokenize(text),
            })

        self._compute_idf()

    def _compute_idf(self):
        """Предподсчёт IDF."""
        N = len(self._chunks)
        if N == 0:
            self.idf = {}
            self.avgdl = 0
            return

        df: dict[str, int] = {}
        total_len = 0
        for chunk in self._chunks:
            seen = set(chunk["tokens"])
            total_len += len(chunk["tokens"])
            for t in seen:
                df[t] = df.get(t, 0) + 1

        self.avgdl = total_len / N
        self.idf = {
            t: math.log((N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 поиск по запросу."""
        if not self._chunks:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = []
        for chunk in self._chunks:
            score = 0.0
            doc_len = len(chunk["tokens"])
            freq: dict[str, int] = {}
            for t in chunk["tokens"]:
                freq[t] = freq.get(t, 0) + 1

            for t in q_tokens:
                if t not in freq:
                    continue
                idf = self.idf.get(t, 0)
                f = freq[t]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        # Top-k
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, sc in indexed[:top_k]:
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "score": sc,
            })
        return results
