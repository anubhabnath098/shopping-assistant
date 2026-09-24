from abc import ABC, abstractmethod
from typing import List
from core.models.schemas import Document, Chunk


class BaseChunker(ABC):
    """Contract for splitting a Document into indexable Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        raise NotImplementedError