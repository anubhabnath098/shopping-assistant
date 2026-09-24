from typing import Dict, Type
from core.interfaces.vector_store import BaseVectorStore
from vector_stores.faiss_store import FAISSVectorStore


class VectorStoreFactory:
    _registry: Dict[str, Type[BaseVectorStore]] = {
        "faiss": FAISSVectorStore,
    }

    @classmethod
    def register(cls, key: str, store_cls: Type[BaseVectorStore]) -> None:
        cls._registry[key] = store_cls

    @classmethod
    def create(cls, store_type: str, dimension: int, index_type: str = "flat") -> BaseVectorStore:
        store_cls = cls._registry.get(store_type.lower())
        if store_cls is None:
            raise ValueError(f"Unknown vector store type: {store_type}. Available: {list(cls._registry)}")
        return store_cls(dimension=int(dimension), index_type=index_type)