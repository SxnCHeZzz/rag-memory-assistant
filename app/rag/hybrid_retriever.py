"""Hybrid retriever: dense (vector) + sparse (BM25) — production-grade.

Использует Reciprocal Rank Fusion (RRF) для объединения результатов.
RRF не зависит от шкал scores (cosine vs BM25), только от рангов.
"""

import logging

from app.config import settings
from app.embeddings.embedder import generate_embeddings
from app.vector.qdrant_client import QdrantService
from app.rag.bm25_index import BM25Index

logger = logging.getLogger("retrieval")


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion: 1 / (k + rank)."""
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantService | None = None,
        bm25: BM25Index | None = None,
    ):
        self.qdrant = qdrant or QdrantService()
        self.bm25 = bm25 or BM25Index()
        self.threshold = settings.retrieval_score_threshold
        self.bm25_weight = settings.bm25_weight
        self.vector_weight = settings.vector_weight
        self.hybrid_enabled = settings.hybrid_search_enabled

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: объединяет vector similarity + BM25 keyword search."""
        # 1. Vector search
        embeddings = generate_embeddings([query], is_query=True)
        vector_results = self.qdrant.search(vector=embeddings[0], top_k=top_k * 2)
        vector_results = [r for r in vector_results if r.score >= self.threshold]

        # 2. BM25 search
        if self.hybrid_enabled:
            bm25_results = self.bm25.search(query, top_k=top_k * 2)
        else:
            bm25_results = []

        # 3. RRF Fusion
        merged = self._rrf_fusion(vector_results, bm25_results, top_k)

        logger.info(
            "query=%r raw_vector=%d raw_bm25=%d final=%d threshold=%.2f",
            query,
            len(vector_results),
            len(bm25_results),
            len(merged),
            self.threshold,
        )
        return merged

    def _rrf_fusion(self, vector_hits, bm25_hits, top_k: int) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        Для каждого источника (vector, BM25) чем выше rank, тем больше score.
        RRF score = weight * (1 / (k + rank)).
        """
        scores: dict[str, dict] = {}

        # Vector: rank 1, 2, 3, ...
        for rank, hit in enumerate(vector_hits, start=1):
            uid = f"{hit.payload.get('source_file', '')}::{hit.payload.get('chunk_index', 0)}"
            rrf = _rrf_score(rank) * self.vector_weight
            scores[uid] = {
                "text": hit.payload.get("text", ""),
                "source_file": hit.payload.get("source_file", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": rrf,
                "vector_score": round(hit.score, 4),
                "bm25_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": None,
                "source": "vector",
            }

        # BM25
        if bm25_hits:
            for rank, hit in enumerate(bm25_hits, start=1):
                uid = f"{hit['source_file']}::{hit['chunk_index']}"
                rrf = _rrf_score(rank) * self.bm25_weight
                if uid in scores:
                    scores[uid]["score"] += rrf
                    scores[uid]["bm25_score"] = round(hit["score"], 4)
                    scores[uid]["bm25_rank"] = rank
                    scores[uid]["source"] = "hybrid"
                else:
                    scores[uid] = {
                        "text": hit["text"],
                        "source_file": hit["source_file"],
                        "chunk_index": hit["chunk_index"],
                        "score": rrf,
                        "vector_score": 0.0,
                        "bm25_score": round(hit["score"], 4),
                        "vector_rank": None,
                        "bm25_rank": rank,
                        "source": "bm25",
                    }

        # Filter by threshold, sort, top_k
        results = [
            {
                "score": round(v["score"], 4),
                "text": v["text"],
                "source_file": v["source_file"],
                "chunk_index": v["chunk_index"],
                "vector_score": v["vector_score"],
                "bm25_score": v["bm25_score"],
                "source": v["source"],
            }
            for v in scores.values()
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
