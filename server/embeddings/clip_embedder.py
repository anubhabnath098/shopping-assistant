from typing import List
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from core.interfaces.embedder import BaseImageEmbedder


class CLIPImageEmbedder(BaseImageEmbedder):
    """
    Visual embedder (VisRAG-style). Embeds product/page images and text
    queries into the SAME vector space, so text queries can retrieve images
    directly without OCR.
    """

    DEFAULT_MODEL = "clip-ViT-B-32"

    def __init__(self, model_name: str = None, device: str = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = SentenceTransformer(self._model_name, device=device)
        self._dimension = self._resolve_dimension()

    def _resolve_dimension(self) -> int:
        """
        SentenceTransformer's built-in get_sentence_embedding_dimension()
        returns None for several CLIP builds because it can't introspect the
        CLIP module the same way it does text transformer modules. Fall back
        to actually encoding a throwaway input and reading the output shape.
        """
        reported_dim = self._model.get_sentence_embedding_dimension()
        if reported_dim is not None:
            return int(reported_dim)

        dummy_image = Image.new("RGB", (32, 32), color=(255, 255, 255))
        probe_embedding = self._model.encode(
            [dummy_image], convert_to_numpy=True, show_progress_bar=False
        )
        return int(probe_embedding.shape[1])

    def embed(self, inputs: List[str]) -> np.ndarray:
        """inputs: list of image file paths."""
        if not inputs:
            return np.zeros((0, self._dimension), dtype="float32")
        images = [Image.open(p).convert("RGB") for p in inputs]
        embeddings = self._model.encode(
            images, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.astype("float32")

    def embed_text_query(self, query: str) -> np.ndarray:
        embedding = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"CLIPImageEmbedder({self._model_name})"