"""Генерация embeddings через sentence-transformers (оптимизировано для low VRAM)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def get_dimension() -> int:
    model = _get_model()
    return model.get_sentence_embedding_dimension()


def generate_embeddings(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Генерирует embeddings для списка текстов.

    Для E5 моделей использует префикс 'query: ' для запросов,
    'passage: ' для документов.

    Batch size ограничен для low VRAM (8GB).
    """
    model = _get_model()

    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]

    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
    )
    return [emb.tolist() for emb in embeddings]
