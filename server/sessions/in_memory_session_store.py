import threading
import uuid
from collections import defaultdict, deque
from typing import Dict, Deque, List
from core.interfaces.session import BaseSessionStore
from core.models.schemas import ConversationTurn


class InMemorySessionStore(BaseSessionStore):
    """
    Keeps a bounded rolling window of conversation turns per session_id, in
    process memory. Thread-safe: the WebSocket layer runs several queries for
    the same session concurrently on worker threads (e.g. a slow image query
    plus fast text queries). Swap for a Redis/DB-backed store later by
    implementing the same BaseSessionStore interface — nothing else changes.
    """

    def __init__(self, max_turns: int = 6):
        self._max_turns = max_turns
        self._lock = threading.RLock()
        self._sessions: Dict[str, Deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=self._max_turns)
        )

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[session_id]  # touch to initialize the deque
        return session_id

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self._sessions[session_id].append(turn)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)