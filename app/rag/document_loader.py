"""Загрузчик документов из текстовых и PDF файлов (production-grade, encoding-safe)."""

import logging
import re
from pathlib import Path
from typing import Iterator

from PyPDF2 import PdfReader

logger = logging.getLogger("documents")


def _fix_mojibake(text: str) -> str:
    """
    Пытается исправить UTF-8 bytes, ошибочно прочитанные как Latin-1/cp1252.

    Если text содержит последовательности типа 'Ð\x9fÑ\x80...' —
    это UTF-8 bytes прочтённые как cp1252.
    """
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        logger.debug("Fixed mojibake: text was UTF-8 misread as Latin-1")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _looks_like_mojibake(text: str) -> bool:
    """
    Херистика: если в тексте много 'Ð', 'Ñ', 'â' подряд — скорее всего мojibake.
    """
    return "Ð" in text or "Ñ" in text or "â\x80" in text


def clean_text(text: str) -> str:
    """Базовая очистка текста + mojibake recovery."""
    text = text.replace("\x00", "")  # null bytes
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Auto-fix mojibake if detected
    if _looks_like_mojibake(text):
        text = _fix_mojibake(text)

    return text.strip()


def load_txt(path: Path) -> str:
    """
    Читает .txt с явным UTF-8 и BOM-handling.

    encoding="utf-8-sig" — автоматически убирает UTF-8 BOM если есть.
    errors="replace" — не упадёт на повреждённых байтах.
    """
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    return clean_text(raw)


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return clean_text("\n".join(parts))


def iter_documents(directory: Path | str = "data/documents") -> Iterator[dict]:
    """Перебирает документы в указанной директории."""
    directory = Path(directory)
    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.suffix.lower() == ".txt":
            text = load_txt(path)
        elif path.suffix.lower() == ".pdf":
            text = load_pdf(path)
        else:
            continue

        if not text:
            continue

        # Sanity check: log if we still see mojibake
        if _looks_like_mojibake(text):
            logger.warning(
                "File %s still contains mojibake after loading! "
                "Check source file encoding.",
                path.name,
            )

        yield {
            "source_file": path.name,
            "text": text,
            "character_count": len(text),
        }
