"""
Shared data contracts used across every layer of the pipeline.
Keeping these in one place avoids tight coupling between modules —
every component talks in terms of these dataclasses, never in
terms of another module's internals.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


@dataclass
class Document:
    """A single page extracted from a source PDF."""
    doc_id: str
    source_path: str
    page_number: int
    text: str
    image_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexable unit — either a text window or a single image."""
    chunk_id: str
    doc_id: str
    source_path: str
    page_number: int
    modality: str  # "text" | "image"
    content: str   # raw text OR filesystem path to an image
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned by some retriever, tagged with its provenance."""
    chunk: Chunk
    score: float
    source_retriever: str  # e.g. "visual_text", "visual_image_via_text", "bm25", "reranked"


class RetrievalPath(Enum):
    LIGHT = "light"
    HEAVY = "heavy"


@dataclass
class RouteDecision:
    """Output of the AdaRAG-style adaptive router."""
    path: RetrievalPath
    confidence: float
    reason: str


@dataclass
class MultiModalQuery:
    """
    A single user query, which may carry text only, an image only, or both.
    session_id ties this query to a conversation's rolling history.
    """
    session_id: str
    text: Optional[str] = None
    image_path: Optional[str] = None

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)

    def display_text(self) -> str:
        """Human/log-friendly representation — also used as history fallback."""
        if self.has_text:
            return self.text
        if self.has_image:
            return f"[image query: {self.image_path}]"
        return ""


@dataclass
class ConversationTurn:
    """One completed exchange, stored in session history for follow-up context."""
    query_text: str
    query_image_path: Optional[str]
    answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Final response returned to a non-streaming client (e.g. REST)."""
    session_id: str
    query: str
    answer: str
    route_taken: str
    confidence_score: float
    retrieved_chunks: List[RetrievedChunk]
    latency_seconds: float


@dataclass
class StreamEvent:
    """
    A single event emitted while answering a query in streaming mode.
    type is one of: "status" | "token" | "final" | "error".
    payload is a plain JSON-serializable dict specific to that type.
    """
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)