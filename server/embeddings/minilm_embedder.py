from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from core.interfaces.embedder import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):
    """Textual embedder used to build/query the text FAISS index."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None, device: str = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = SentenceTransformer(self._model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, inputs: List[str]) -> np.ndarray:
        if not inputs:
            return np.zeros((0, self._dimension), dtype="float32")
        embeddings = self._model.encode(
            inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"MiniLMEmbedder({self._model_name})"