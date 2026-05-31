"""Session-based conversation memory."""

from collections import defaultdict
from datetime import datetime

from app.config import settings


class ConversationMemoryService:
    """
    In-memory хранение истории разговоров по session_id.
    Для MVP достаточно. В production — Redis / DB.
    """

    def __init__(self):
        self._histories: dict[str, list[dict]] = defaultdict(list)
        self._summaries: dict[str, str] = {}

    def add_turn(self, session_id: str, role: str, content: str):
        """Добавить сообщение в историю."""
        self._histories[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._trim(session_id)

    def get_context(self, session_id: str) -> tuple[list[str], str | None]:
        """
        Возвращает (recent_messages, summary_or_none).
        recent_messages — последние N сообщений.
        summary — если история длинная.
        """
        history = self._histories[session_id]
        if not history:
            return [], None

        max_len = settings.conversation_history_max
        threshold = settings.conversation_summary_threshold

        if len(history) <= max_len:
            return [f"{h['role']}: {h['content']}" for h in history], None

        # Если длиннее threshold — вернуть summary + последние 2
        summary = self._summarize(session_id)
        recent = history[-2:]
        msgs = [f"{h['role']}: {h['content']}" for h in recent]
        return msgs, summary

    def _trim(self, session_id: str):
        """Обрезает историю до max_len."""
        max_len = settings.conversation_history_max
        history = self._histories[session_id]
        if len(history) > max_len * 2:
            # Summarize old messages
            self._summarize(session_id)
            # Keep last max_len
            self._histories[session_id] = history[-max_len:]

    def _summarize(self, session_id: str) -> str:
        """Простое правило-based summarization. Можно заменить на LLM summarizer."""
        history = self._histories[session_id]
        # Extract all user messages as a simple summary
        user_msgs = [h["content"] for h in history if h["role"] == "user"]
        if not user_msgs:
            return ""
        summary = "; ".join(user_msgs[:3])
        if len(user_msgs) > 3:
            summary += f" (+{len(user_msgs) - 3} more messages)"
        self._summaries[session_id] = summary
        return summary

    def clear(self, session_id: str):
        self._histories[session_id].clear()
        self._summaries.pop(session_id, None)
