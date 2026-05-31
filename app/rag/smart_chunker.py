"""Sentence-aware chunking с code/markdown awareness."""

import re
from typing import Iterator


RE_SENTENCE = re.compile(r"(?<=[.!?\n])\s+(?=[А-ЯЁA-Z])")
RE_HEADING = re.compile(r"^(#{1,6}\s|```|\*\*\*|---)", re.MULTILINE)


def chunk_text_smart(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128,
    source_file: str = "",
) -> Iterator[dict]:
    """
    Разбивает текст на чанки, уважая границы предложений.

    Стратегия:
    1. Если текст короткий — один чанк
    2. Ищем конец предложения ближе к chunk_size
    3. Если в чанке код — не разрывать блоки ```
    """
    text_len = len(text)
    if text_len <= chunk_size:
        yield _make_chunk(text, source_file, 0, text)
        return

    step = chunk_size - overlap
    start = 0
    index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Пытаемся отступить до конца предложения
        if end < text_len:
            # Ищем ближайший перенос строки или конец предложения
            search_start = max(start + chunk_size // 2, end - 100)
            region = text[search_start:end]
            # Ищем перенос строки
            lb = region.rfind("\n")
            if lb >= 0:
                end = search_start + lb + 1
            else:
                # Ищем конец предложения
                found = RE_SENTENCE.findall(region)
                if found:
                    # Найти последнее совпадение
                    pos = region.rfind(found[-1])
                    end = search_start + pos + len(found[-1])

        chunk = text[start:end]
        yield _make_chunk(chunk, source_file, index, text)

        # Следующий chunk
        advance = end - overlap
        if advance <= start:
            advance = end
        start = advance
        index += 1

        if end >= text_len:
            break


def _make_chunk(text: str, source_file: str, index: int, full_text: str) -> dict:
    """Создаёт чанк с metadata."""
    text_stripped = text.strip()
    has_code = "```" in text_stripped
    is_heading = bool(RE_HEADING.search(text_stripped))

    # Определяем type
    if has_code:
        chunk_type = "code"
    elif is_heading:
        chunk_type = "heading"
    elif text_stripped.startswith("-") or text_stripped.startswith("*"):
        chunk_type = "list"
    else:
        chunk_type = "text"

    # Определяем язык кода
    lang = None
    if has_code:
        m = re.search(r"```(\w+)", text_stripped)
        if m:
            lang = m.group(1)

    return {
        "text": text_stripped,
        "source_file": source_file,
        "chunk_index": index,
        "character_count": len(text_stripped),
        "chunk_type": chunk_type,
        "has_code": has_code,
        "language": lang,
    }
