"""Сборка prompt с retrieved documents, user memory и conversation context."""

from app.config import settings


SYSTEM_PROMPT = (
    "Ты полезный ассистент. Отвечай ТОЛЬКО на основе предоставленного контекста. "
    "Если в контексте нет ответа — честно скажи, что не знаешь. "
    "Не придумывай факты, которых нет в контексте. "
    "Давай полный ответ, если контекст содержит достаточную информацию."
)

MAX_PROMPT_SIZE = 1800  # safety limit


def _compress_chunks(chunks: list[dict], max_chars: int) -> list[dict]:
    """Обрезает чанки если общий prompt слишком длинный."""
    total = sum(len(c["text"]) for c in chunks)
    if total <= max_chars:
        return chunks

    
    seen: set[str] = set()
    unique = []
    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

  
    total_u = sum(len(c["text"]) for c in unique)
    if total_u <= max_chars:
        return unique

    unique.sort(key=lambda x: x.get("score", 0), reverse=True)
    result = []
    current = 0
    for c in unique:
        if current + len(c["text"]) > max_chars:
            break
        result.append(c)
        current += len(c["text"])
    return result


def build_prompt(
    question: str,
    documents: list[dict],
    memories: list[dict],
    conversation_messages: list[str] | None = None,
    conversation_summary: str | None = None,
) -> tuple[str, str]:
    """
    Собирает system prompt и user prompt для LLM.

    Возвращает (system, prompt).
    """
    overhead = len(SYSTEM_PROMPT) + len(question) + 500  # запас под memories + conversation
    documents = _compress_chunks(documents, max(500, MAX_PROMPT_SIZE - overhead))       

    parts = []

    # Conversation context
    if conversation_summary:
        parts.append("=== История разговора (сводка) ===")
        parts.append(conversation_summary)
        parts.append("")

    if conversation_messages:
        parts.append("=== Недавние сообщения ===")
        for msg in conversation_messages:
            parts.append(msg)
        parts.append("")

    # User memory
    if memories:
        parts.append("=== Память пользователя ===")
        for mem in memories:
            parts.append(f"- {mem['text']}")
        parts.append("")

    # Documents
    if documents:
        parts.append("=== Контекст из документов ===")
        for doc in documents:
            source = f"[Источник: {doc['source_file']}"
            if doc.get("chunk_type") == "code":
                lang = doc.get("language", "")
                source += f", code: {lang}" if lang else ", code"
            source += "]"
            parts.append(f"\n{source}\n{doc['text']}\n")
        parts.append("---")
        parts.append("")

    parts.append(f"Вопрос: {question}")
    parts.append("\nОтвет (только на основе контекста):")

    prompt = "\n".join(parts)
    return SYSTEM_PROMPT, prompt
