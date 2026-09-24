import os
import pickle
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from core.interfaces.retriever import BaseRetriever
from core.models.schemas import Chunk, RetrievedChunk, MultiModalQuery


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25CorpusStore:
    """Persists/loads the raw text-chunk corpus that BM25 is built from."""

    def __init__(self, corpus_path: str):
        self._corpus_path = corpus_path

    def save(self, chunks: List[Chunk]) -> None:
        os.makedirs(os.path.dirname(self._corpus_path), exist_ok=True)
        payload = [
            {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "source_path": c.source_path,
                "page_number": c.page_number, "modality": c.modality, "content": c.content,
                "metadata": c.metadata,
            }
            for c in chunks if c.modality == "text"
        ]
        with open(self._corpus_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._corpus_path):
            return []
        with open(self._corpus_path, "rb") as f:
            return pickle.load(f)


class BM25Retriever(BaseRetriever):
    """Sparse lexical retriever — the keyword-based half of the AdaRAG heavy path."""

    def __init__(self, corpus_store: BM25CorpusStore):
        self._corpus_store = corpus_store
        self._corpus: List[Dict[str, Any]] = []
        self._bm25 = None
        self._load()

    def _load(self) -> None:
        self._corpus = self._corpus_store.load()
        tokenized = [_tokenize(item["content"]) for item in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, query: MultiModalQuery, top_k: int) -> List[RetrievedChunk]:
        # BM25 is a lexical retriever — it has nothing to do with an image-only query.
        if not self._bm25 or not self._corpus or not query.has_text:
            return []
        scores = self._bm25.get_scores(_tokenize(query.text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked_indices:
            item = self._corpus[idx]
            chunk = Chunk(
                chunk_id=item["chunk_id"], doc_id=item["doc_id"], source_path=item["source_path"],
                page_number=item["page_number"], modality=item["modality"], content=item["content"],
                metadata=item["metadata"],
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(scores[idx]), source_retriever="bm25"))
        return results