"""
Embedding abstractions. Any new embedding model (text or image) just needs
to implement one of these interfaces and register itself with a factory —
nothing else in the codebase needs to change (Open/Closed Principle).
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Contract for any embedder that turns inputs into vectors."""

    @abstractmethod
    def embed(self, inputs: List[str]) -> np.ndarray:
        """Embed a batch of inputs, returning an (N, dim) float32 array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class BaseImageEmbedder(BaseEmbedder):
    """
    Cross-modal embedder: embed() encodes images into the shared space,
    embed_text_query() encodes a text query into that SAME space, so a text
    query can retrieve directly against the image index (VisRAG-style).
    """

    @abstractmethod
    def embed_text_query(self, query: str) -> np.ndarray:
        raise NotImplementedError