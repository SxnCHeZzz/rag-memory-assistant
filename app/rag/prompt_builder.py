"""Сборка prompt с retrieved documents, user memory и conversation context."""

from app.config import settings

# ИСПРАВЛЕНО: Явно разрешаем использовать Память пользователя и Историю бесед
SYSTEM_PROMPT = (
    "Ты — умный локальный ИИ-ассистент. Твоя главная цель — помогать пользователю, основываясь ТОЛЬКО на предоставленных данных.\n"
    "У тебя есть доступ к трем блокам информации:\n"
    "1. ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ (Факты о собеседнике: его имя, учеба, увлечения, хобби).\n"
    "2. КОНТЕКСТ ИЗ ДОКУМЕНТОВ (Внешние справочные файлы и базы знаний).\n"
    "3. ИСТОРИЯ ТЕКУЩЕЙ БЕСЕДЫ (Контекст диалога).\n\n"
    "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
    "- Если в блоке 'ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ' написано, что данные не сохранены, и этот блок пуст — ты ЗАПРЕЩАЕШЬ себе угадывать, придумывать или домысливать имя, учебу или хобби пользователя. В этом случае на любые вопросы вроде 'как меня зовут?' или 'где я учусь?' ты обязан ответить строго: 'У меня нет информации о вашей личности в базе данных. Пожалуйста, сохраните её через эндпоинт /memory'.\n"
    "- Отвечай СТРОГО на основе предоставленной информации из этих трех источников.\n"
    "- Не выдумывай факты от себя. При обращении к пользователю учитывай данные из его памяти."
)

MAX_PROMPT_SIZE = 8000


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

    parts.append(f"Вопрос пользователя: {question}")
    parts.append("\nОтвет (на основе памяти и документов):")

    prompt = "\n".join(parts)
    return SYSTEM_PROMPT, prompt