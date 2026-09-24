from abc import ABC, abstractmethod
from typing import List
from core.models.schemas import RetrievedChunk, MultiModalQuery


class BaseReranker(ABC):
    """Contract for fusing/reordering a pool of candidates into a final ranking."""

    @abstractmethod
    def rerank(self, query: MultiModalQuery, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError