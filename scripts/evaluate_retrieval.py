"""Evaluation script для сравнения качества retrieval.

Сравнивает:
- vector only
- hybrid (vector + BM25)
- reranked

Выводит таблицу с scores.
"""

import time
from pathlib import Path

from app.config import settings
from app.vector.qdrant_client import QdrantService
from app.rag.hybrid_retriever import HybridRetriever
from app.embeddings.embedder import generate_embeddings


TEST_QUERIES = [
    ("что такое embeddings", ["embeddings.txt"]),
    ("FastAPI преимущества", ["fastapi.txt"]),
    ("Qdrant как база данных", ["vector_databases.txt"]),
    ("RAG pipeline", ["rag.txt"]),
    ("memory как хранить предпочтения", ["memory.txt"]),
]


def check_relevance(results: list[dict], expected_files: list[str]) -> dict:
    """Оценивает релевантность: precision @ top_k."""
    if not results:
        return {"precision@1": 0.0, "precision@3": 0.0, "mrr": 0.0}

    # Precision @ 1
    p1 = 1.0 if results[0]["source_file"] in expected_files else 0.0

    # Precision @ 3
    top3 = [r["source_file"] in expected_files for r in results[:3]]
    p3 = sum(top3) / len(top3) if top3 else 0.0

    # MRR
    mrr = 0.0
    for i, r in enumerate(results):
        if r["source_file"] in expected_files:
            mrr = 1.0 / (i + 1)
            break

    return {"precision@1": p1, "precision@3": p3, "mrr": mrr}


def evaluate_vector_only(qdrant: QdrantService, query: str, top_k: int = 5):
    embeddings = generate_embeddings([query], is_query=True)
    results = qdrant.search(vector=embeddings[0], top_k=top_k)
    return [
        {
            "score": r.score,
            "text": r.payload.get("text", "")[:50],
            "source_file": r.payload.get("source_file", ""),
        }
        for r in results
        if r.score >= settings.retrieval_score_threshold
    ]


def main():
    print("=" * 80)
    print("Retrieval Quality Evaluation")
    print("=" * 80)

    qdrant = QdrantService()
    hybrid = HybridRetriever(qdrant=qdrant)

    vector_scores = []
    hybrid_scores = []

    for query, expected in TEST_QUERIES:
        print(f"\nQuery: {query}")

        # Vector only
        t0 = time.time()
        vec_results = evaluate_vector_only(qdrant, query)
        v_metrics = check_relevance(vec_results, expected)
        vector_scores.append(v_metrics)
        print(f"  Vector    P@1={v_metrics['precision@1']:.2f}  P@3={v_metrics['precision@3']:.2f}  MRR={v_metrics['mrr']:.2f}  ({len(vec_results)} results, {(time.time()-t0)*1000:.0f}ms)")

        # Hybrid
        t0 = time.time()
        hy_results = hybrid.retrieve(query, top_k=5)
        h_metrics = check_relevance(hy_results, expected)
        hybrid_scores.append(h_metrics)
        print(f"  Hybrid    P@1={h_metrics['precision@1']:.2f}  P@3={h_metrics['precision@3']:.2f}  MRR={h_metrics['mrr']:.2f}  ({len(hy_results)} results, {(time.time()-t0)*1000:.0f}ms)")

    # Averages
    def avg(metrics_list, key):
        return sum(m[key] for m in metrics_list) / len(metrics_list)

    print("\n" + "=" * 80)
    print("Averages")
    print("-" * 80)
    print(f"  Vector   P@1={avg(vector_scores, 'precision@1'):.3f}  P@3={avg(vector_scores, 'precision@3'):.3f}  MRR={avg(vector_scores, 'mrr'):.3f}")
    print(f"  Hybrid   P@1={avg(hybrid_scores, 'precision@1'):.3f}  P@3={avg(hybrid_scores, 'precision@3'):.3f}  MRR={avg(hybrid_scores, 'mrr'):.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
