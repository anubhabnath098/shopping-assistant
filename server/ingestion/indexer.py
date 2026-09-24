import os
import glob
from typing import List
from core.interfaces.document_loader import BaseDocumentLoader
from core.interfaces.chunker import BaseChunker
from core.interfaces.embedder import BaseEmbedder, BaseImageEmbedder
from core.interfaces.vector_store import BaseVectorStore
from core.models.schemas import Chunk


class IndexBuilder:
    """
    Orchestrates the offline 'data preparation' stage:
    PDFs -> Documents -> Chunks -> Embeddings -> Vector stores + BM25 corpus.
    """

    def __init__(
        self,
        document_loader: BaseDocumentLoader,
        text_chunker: BaseChunker,
        image_chunker: BaseChunker,
        text_embedder: BaseEmbedder,
        image_embedder: BaseImageEmbedder,
        text_store: BaseVectorStore,
        image_store: BaseVectorStore,
        bm25_corpus_writer,  # callable(List[Chunk]) -> None
    ):
        self._document_loader = document_loader
        self._text_chunker = text_chunker
        self._image_chunker = image_chunker
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._text_store = text_store
        self._image_store = image_store
        self._bm25_corpus_writer = bm25_corpus_writer

    def build_from_folder(self, folder_path: str) -> None:
        pdf_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True))
        if not pdf_files:
            print(f"No PDF files found under: {folder_path}")
            return

        all_text_chunks: List[Chunk] = []
        all_image_chunks: List[Chunk] = []

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path}")
            for document in self._document_loader.load(pdf_path):
                all_text_chunks.extend(self._text_chunker.chunk(document))
                all_image_chunks.extend(self._image_chunker.chunk(document))

        print(f"Total text chunks: {len(all_text_chunks)} | Total image chunks: {len(all_image_chunks)}")

        if all_text_chunks:
            vectors = self._text_embedder.embed([c.content for c in all_text_chunks])
            self._text_store.add(vectors, [self._chunk_to_metadata(c) for c in all_text_chunks])

        if all_image_chunks:
            vectors = self._image_embedder.embed([c.content for c in all_image_chunks])
            self._image_store.add(vectors, [self._chunk_to_metadata(c) for c in all_image_chunks])

        self._bm25_corpus_writer(all_text_chunks)
        print("Indexing complete.")

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "source_path": chunk.source_path,
            "page_number": chunk.page_number, "modality": chunk.modality, "content": chunk.content,
            "metadata": chunk.metadata,
        }