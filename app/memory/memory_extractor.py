"""Простой rule-based extractor для извлечения memory из текста."""

import re

# Простые паттерны, которые сигнализируют о предпочтениях пользователя
PATTERNS = [
    r"я\s+использую\s+(.+)",
    r"мой\s+проект\s+(.+)",
    r"я\s+предпочитаю\s+(.+)",
    r"я\s+работаю\s+с\s+(.+)",
    r"мой\s+любимый\s+(.+)",
    r"я\s+не\s+люблю\s+(.+)",
    r"я\s+хочу\s+(.+)",
    r"я\s+изучаю\s+(.+)",
]

KEYWORDS = [
    "python",
    "fastapi",
    "ollama",
    "qdrant",
    "pytorch",
    "tensorflow",
    "rust",
    "go",
    "typescript",
    "javascript",
    "react",
    "vue",
]


def extract_memories(text: str) -> list[tuple[str, str]]:
    """
    Извлекает потенциальные memory-записи из текста.

    Возвращает список (category, extracted_text).
    """
    found = []
    lower = text.lower()

    for pattern in PATTERNS:
        for match in re.finditer(pattern, lower):
            extracted = match.group(1).strip().rstrip(".!?,;")
            if len(extracted) > 2:
                found.append(("preference", extracted))

    for kw in KEYWORDS:
        if kw in lower:
            found.append(("technology", kw))

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique = []
    for category, item in found:
        key = (category, item)
        if key not in seen:
            seen.add(key)
            unique.append((category, item))

    return unique
