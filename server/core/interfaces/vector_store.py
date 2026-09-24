from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseVectorStore(ABC):
    """Contract for any vector database backend (FAISS, Milvus, Pinecone, ...)."""

    @abstractmethod
    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Returns a list of (similarity_score, metadata) sorted best-first."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError