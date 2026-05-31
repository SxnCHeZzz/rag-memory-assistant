"""Optional reranker (cross-encoder) для улучшения retrieval quality.

ВАЖНО: lazy-load модели чтобы не замедлять старт.
Работает на CPU чтобы экономить VRAM.
"""

import logging

from app.config import settings

logger = logging.getLogger("reranker")


class RerankerClient:
    """Lazy-loaded cross-encoder reranker."""

    def __init__(self):
        self._model = None
        self.enabled = settings.reranker_enabled
        self.model_name = settings.reranker_model
        self.batch_size = settings.reranker_batch_size
        self.top_k_before = settings.reranker_top_k_before
        self.top_k_after = settings.reranker_top_k_after

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device="cpu",
            )
            logger.info("Reranker loaded OK")
        except Exception as e:
            logger.error("Failed to load reranker: %s", e)
            self.enabled = False

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Принимает чанки, возвращает top-k после reranking."""
        if not self.enabled or not chunks:
            return chunks

        self._load()
        if self._model is None:
            return chunks

        # Prepare pairs
        pairs = [[query, c["text"]] for c in chunks]

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach rerank scores
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = round(float(scores[i]), 4)

        # Sort by rerank score
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:self.top_k_after]
