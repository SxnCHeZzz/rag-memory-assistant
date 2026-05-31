"""Простой chunking sliding window."""

from typing import Iterator

from app.config import settings
from app.rag.smart_chunker import chunk_text_smart


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128,
    source_file: str = "",
) -> Iterator[dict]:
    """Разбивает текст на чанки с перекрытием."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")

    # Используем smart chunking с sentence awareness
    yield from chunk_text_smart(text, chunk_size, overlap, source_file)
