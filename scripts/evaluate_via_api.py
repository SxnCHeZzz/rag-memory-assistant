"""Retrieval evaluation via HTTP API (no Qdrant file lock needed).

Runs against the live /debug/retrieval endpoint and computes:
- Precision@1, Precision@3
- MRR (Mean Reciprocal Rank)
"""

import json
import time
import sys
import urllib.request

# Force UTF-8 on Windows to avoid cp1251/cp866 mojibake
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

API_URL = "http://127.0.0.1:8000/debug/retrieval"

TEST_QUERIES = [
    ("что такое embeddings", ["embeddings.txt", "embeddings_guide.txt"]),
    ("FastAPI преимущества", ["fastapi.txt", "fastapi_guide.txt"]),
    ("Qdrant как база данных", ["vector_databases.txt", "qdrant.txt"]),
    ("RAG pipeline", ["rag.txt", "README.txt"]),
    ("memory как хранить предпочтения", ["memory.txt", "conversation.txt"]),
]


def retrieve_via_api(query: str, top_k: int = 5):
    payload = json.dumps({"question": query, "top_k": top_k}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode("utf-8"))


def check_relevance(results: list[dict], expected_files: list[str]) -> dict:
    if not results:
        return {"precision@1": 0.0, "precision@3": 0.0, "mrr": 0.0}

    p1 = 1.0 if results[0]["source_file"] in expected_files else 0.0

    top3 = [r["source_file"] in expected_files for r in results[:3]]
    p3 = sum(top3) / len(top3) if top3 else 0.0

    mrr = 0.0
    for i, r in enumerate(results):
        if r["source_file"] in expected_files:
            mrr = 1.0 / (i + 1)
            break

    return {"precision@1": p1, "precision@3": p3, "mrr": mrr}


def main():
    print("=" * 80)
    print("Retrieval Quality Evaluation (via HTTP API)")
    print("=" * 80)

    all_scores = []

    for query, expected in TEST_QUERIES:
        t0 = time.time()
        resp = retrieve_via_api(query, top_k=5)
        docs = resp.get("documents", [])
        elapsed = (time.time() - t0) * 1000

        metrics = check_relevance(docs, expected)
        all_scores.append(metrics)

        hybrid = resp.get("hybrid_enabled", False)
        reranker = resp.get("reranker_enabled", False)

        print(f"\nQuery: {query}")
        print(f"  Mode: hybrid={hybrid}, reranker={reranker}")
        print(f"  Docs retrieved: {len(docs)} ({elapsed:.0f}ms)")
        for i, d in enumerate(docs[:3], 1):
            hit = "[OK]" if d['source_file'] in expected else "[NO]"
            print(f"  {i}. {hit} score={d['score']:.4f} [{d['source']}] {d['source_file']}")
        print(f"     P@1={metrics['precision@1']:.2f}  P@3={metrics['precision@3']:.2f}  MRR={metrics['mrr']:.2f}")

    def avg(metrics_list, key):
        return sum(m[key] for m in metrics_list) / len(metrics_list)

    print("\n" + "=" * 80)
    print("Averages")
    print("-" * 80)
    print(f"  P@1 ={avg(all_scores, 'precision@1'):.3f}")
    print(f"  P@3 ={avg(all_scores, 'precision@3'):.3f}")
    print(f"  MRR ={avg(all_scores, 'mrr'):.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
