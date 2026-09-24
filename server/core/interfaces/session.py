from abc import ABC, abstractmethod
from typing import List
from core.models.schemas import ConversationTurn


class BaseSessionStore(ABC):
    """Contract for storing/retrieving per-session conversation history."""

    @abstractmethod
    def create_session(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, session_id: str) -> List[ConversationTurn]:
        raise NotImplementedError

    @abstractmethod
    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str) -> None:
        raise NotImplementedError