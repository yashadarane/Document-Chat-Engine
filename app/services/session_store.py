from __future__ import annotations

from threading import RLock
from uuid import uuid4

from app.models.domain import ParsedDocument, SessionState
from app.services.memory import ConversationMemory
from app.core.exceptions import SessionNotFoundError


class SessionStore:
    """Thread-safe in-memory session storage."""

    def __init__(self, memory_turns: int) -> None:
        self._memory_turns = memory_turns
        self._sessions: dict[str, SessionState] = {}
        self._lock = RLock()

    def create_session(self, documents: list[ParsedDocument]) -> SessionState:
        session = SessionState(
            session_id=uuid4().hex,
            documents=documents,
            memory=ConversationMemory(max_turns=self._memory_turns),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        return session

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
