"""
Factory Method pattern: to add a new embedding model (e.g. a bigger text
model or a different image encoder), implement the interface and register
it here — the rest of the pipeline never changes.
"""
from typing import Dict, Type
from core.interfaces.embedder import BaseEmbedder, BaseImageEmbedder
from embeddings.minilm_embedder import MiniLMEmbedder
from embeddings.clip_embedder import CLIPImageEmbedder


class TextEmbedderFactory:
    _registry: Dict[str, Type[BaseEmbedder]] = {
        "minilm": MiniLMEmbedder,
    }

    @classmethod
    def register(cls, key: str, embedder_cls: Type[BaseEmbedder]) -> None:
        cls._registry[key] = embedder_cls

    @classmethod
    def create(cls, embedder_type: str, **kwargs) -> BaseEmbedder:
        embedder_cls = cls._registry.get(embedder_type.lower())
        if embedder_cls is None:
            raise ValueError(f"Unknown text embedder type: {embedder_type}. Available: {list(cls._registry)}")
        return embedder_cls(**kwargs)


class ImageEmbedderFactory:
    _registry: Dict[str, Type[BaseImageEmbedder]] = {
        "clip": CLIPImageEmbedder,
    }

    @classmethod
    def register(cls, key: str, embedder_cls: Type[BaseImageEmbedder]) -> None:
        cls._registry[key] = embedder_cls

    @classmethod
    def create(cls, embedder_type: str, **kwargs) -> BaseImageEmbedder:
        embedder_cls = cls._registry.get(embedder_type.lower())
        if embedder_cls is None:
            raise ValueError(f"Unknown image embedder type: {embedder_type}. Available: {list(cls._registry)}")
        return embedder_cls(**kwargs)