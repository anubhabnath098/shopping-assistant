from typing import List
from core.interfaces.retriever import BaseRetriever
from core.interfaces.embedder import BaseEmbedder, BaseImageEmbedder
from core.interfaces.vector_store import BaseVectorStore
from core.models.schemas import Chunk, RetrievedChunk, MultiModalQuery


class DualIndexRetriever(BaseRetriever):
    """
    VisRAG-inspired dense retriever: depending on what the query carries, it
    searches the text index (MiniLM space) with query text, the image index
    (CLIP space) with query text, and/or the image index with a query IMAGE
    (image-to-image search) — then fuses everything into one ranked list.
    """

    def __init__(
        self,
        text_embedder: BaseEmbedder,
        image_embedder: BaseImageEmbedder,
        text_store: BaseVectorStore,
        image_store: BaseVectorStore,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ):
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._text_store = text_store
        self._image_store = image_store
        self._text_weight = text_weight
        self._image_weight = image_weight

    def retrieve(self, query: MultiModalQuery, top_k: int) -> List[RetrievedChunk]:
        merged: List[RetrievedChunk] = []

        if query.has_text:
            merged.extend(self._search_text_index_with_text(query.text, top_k))
            merged.extend(self._search_image_index_with_text(query.text, top_k))

        if query.has_image:
            merged.extend(self._search_image_index_with_image(query.image_path, top_k))

        merged.sort(key=lambda rc: rc.score, reverse=True)
        return self._deduplicate(merged)[:top_k]

    def top_confidence(self, query: MultiModalQuery) -> float:
        """Highest similarity score across modalities — feeds the AdaRAG router."""
        hits = self.retrieve(query, top_k=1)
        return hits[0].score if hits else 0.0

    def text_index_size(self) -> int:
        return self._text_store.size

    def image_index_size(self) -> int:
        return self._image_store.size

    def _search_text_index_with_text(self, text: str, top_k: int) -> List[RetrievedChunk]:
        if self._text_store.size == 0:
            return []
        query_vector = self._text_embedder.embed([text])[0]
        raw_hits = self._text_store.search(query_vector, top_k)
        return [
            RetrievedChunk(chunk=self._metadata_to_chunk(meta), score=score * self._text_weight, source_retriever="visual_text")
            for score, meta in raw_hits
        ]

    def _search_image_index_with_text(self, text: str, top_k: int) -> List[RetrievedChunk]:
        if self._image_store.size == 0:
            return []
        query_vector = self._image_embedder.embed_text_query(text)[0]
        raw_hits = self._image_store.search(query_vector, top_k)
        return [
            RetrievedChunk(chunk=self._metadata_to_chunk(meta), score=score * self._image_weight, source_retriever="visual_image_via_text")
            for score, meta in raw_hits
        ]

    def _search_image_index_with_image(self, image_path: str, top_k: int) -> List[RetrievedChunk]:
        """Image-to-image retrieval in CLIP space — used for image-only or image+text queries."""
        if self._image_store.size == 0:
            return []
        query_vector = self._image_embedder.embed([image_path])[0]
        raw_hits = self._image_store.search(query_vector, top_k)
        return [
            RetrievedChunk(chunk=self._metadata_to_chunk(meta), score=score, source_retriever="visual_image_via_image")
            for score, meta in raw_hits
        ]

    @staticmethod
    def _deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen = set()
        result = []
        for rc in chunks:
            if rc.chunk.chunk_id in seen:
                continue
            seen.add(rc.chunk.chunk_id)
            result.append(rc)
        return result

    @staticmethod
    def _metadata_to_chunk(meta) -> Chunk:
        return Chunk(
            chunk_id=meta["chunk_id"], doc_id=meta["doc_id"], source_path=meta["source_path"],
            page_number=meta["page_number"], modality=meta["modality"], content=meta["content"],
            metadata=meta.get("metadata", {}),
        )