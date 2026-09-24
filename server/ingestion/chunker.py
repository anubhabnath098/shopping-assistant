import uuid
from typing import List
from core.interfaces.chunker import BaseChunker
from core.models.schemas import Document, Chunk


class FixedSizeTextChunker(BaseChunker):
    """Splits a document's text into overlapping fixed-size word windows."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        words = document.text.split()
        if not words:
            return chunks
        step = max(self._chunk_size - self._overlap, 1)
        for start in range(0, len(words), step):
            piece = words[start:start + self._chunk_size]
            if not piece:
                continue
            content = " ".join(piece)
            chunk_id = f"{document.doc_id}_t{start}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    page_number=document.page_number,
                    modality="text",
                    content=content,
                    metadata=dict(document.metadata),
                )
            )
            if start + self._chunk_size >= len(words):
                break
        return chunks


class ImageChunker(BaseChunker):
    """Turns each image extracted from a page into its own retrievable chunk."""

    def chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        for image_path in document.image_paths:
            chunk_id = f"{document.doc_id}_img_{uuid.uuid4().hex[:6]}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    page_number=document.page_number,
                    modality="image",
                    content=image_path,
                    metadata=dict(document.metadata),
                )
            )
        return chunks