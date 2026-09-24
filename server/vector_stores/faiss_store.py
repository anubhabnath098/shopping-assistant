import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from core.interfaces.vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """
    Generic FAISS-backed vector store. Works for both the text index and the
    image index — only the dimension/index_type differ, injected at creation.
    """

    def __init__(self, dimension: int, index_type: str = "flat"):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"FAISSVectorStore requires a positive integer dimension, got: {dimension!r} "
                f"(type={type(dimension).__name__}). This usually means an embedder's "
                f".dimension property returned None or a non-int value."
            )
        self._dimension = dimension
        self._index_type = index_type
        self._index = self._build_index(index_type, dimension)
        self._metadatas: List[Dict[str, Any]] = []

    @staticmethod
    def _build_index(index_type: str, dimension: int):
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 32
            return index
        if index_type == "flat":
            return faiss.IndexFlatIP(dimension)
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if vectors.shape[0] == 0:
            return
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        self._index.add(vectors)
        self._metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if self._index.ntotal == 0:
            return []
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self._metadatas[idx]))
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(
                {"metadatas": self._metadatas, "dimension": self._dimension, "index_type": self._index_type}, f
            )

    def load(self, path: str) -> None:
        self._index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
        self._metadatas = data["metadatas"]
        self._dimension = data["dimension"]
        self._index_type = data["index_type"]

    @property
    def size(self) -> int:
        return self._index.ntotal