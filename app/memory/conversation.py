import logging
from collections import defaultdict
from app.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.sessions: dict[str, list[dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_summary(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        turns = []
        for msg in history[-6:]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{prefix}: {msg['content'][:200]}")
        return "\n".join(turns)


memory = ConversationMemory()
