"""Общий Qdrant клиент для embedded/local mode."""

from pathlib import Path

from qdrant_client import QdrantClient

_DATA_PATH = Path("./qdrant_data")
_DATA_PATH.mkdir(parents=True, exist_ok=True)

_shared_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Возвращает единственный экземпляр QdrantClient."""
    global _shared_client
    if _shared_client is None:
        _shared_client = QdrantClient(path=str(_DATA_PATH), lock=False)
    return _shared_client
