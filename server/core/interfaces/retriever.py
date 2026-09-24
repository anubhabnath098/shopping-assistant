from abc import ABC, abstractmethod
from typing import List
from core.models.schemas import RetrievedChunk, MultiModalQuery


class BaseRetriever(ABC):
    """Contract shared by light, heavy, dense, and sparse retrievers alike."""

    @abstractmethod
    def retrieve(self, query: MultiModalQuery, top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError