import time
from typing import Optional, Iterator, List
from core.models.schemas import (
    RAGResponse, RetrievalPath, MultiModalQuery, ConversationTurn,
    RetrievedChunk, StreamEvent,
)
from generation.generator import ResponseGenerator
from retrieval.light_retriever import LightRetriever
from retrieval.heavy_retriever import HeavyRetriever
from retrieval.router import AdaptiveRouter
from core.interfaces.session import BaseSessionStore
from sessions.query_contextualizer import QueryContextualizer


class RAGPipeline:
    """
    Facade over the full AdaRAG + VisRAG-inspired pipeline:

        query (text/image/both) + session history
              -> follow-up rewritten into a standalone query
              -> light retrieval (confidence check)
              -> AdaRAG router decision
              -> [LIGHT: fast dual-index ANN] or [HEAVY: BM25 + dense + rerank]
              -> LLM generation grounded on retrieved context + history + image
              -> turn saved back into the session

    answer() returns one complete RAGResponse (used by REST/CLI).
    answer_stream() yields StreamEvents as they happen (used by WebSocket),
    so the client can start speaking before generation fully finishes.
    """

    def __init__(
        self,
        light_retriever: LightRetriever,
        heavy_retriever: HeavyRetriever,
        router: AdaptiveRouter,
        generator: ResponseGenerator,
        session_store: BaseSessionStore,
        query_contextualizer: QueryContextualizer,
        top_k: int = 5,
    ):
        self._light_retriever = light_retriever
        self._heavy_retriever = heavy_retriever
        self._router = router
        self._generator = generator
        self._session_store = session_store
        self._query_contextualizer = query_contextualizer
        self._top_k = top_k

    def start_session(self) -> str:
        return self._session_store.create_session()

    def ensure_session(self, session_id: Optional[str]) -> str:
        """
        Central rule used by REST, CLI, and WebSocket alike: reuse the given
        session_id if present, otherwise mint a new one.
        """
        if session_id and session_id.strip():
            return session_id.strip()
        return self.start_session()

    def answer(self, query: MultiModalQuery) -> RAGResponse:
        start_time = time.perf_counter()

        history = self._session_store.get_history(query.session_id)
        resolved_query = self._resolve_query(query, history)
        decision, retrieved_chunks = self._route_and_retrieve(resolved_query)

        answer_text = self._generator.generate(resolved_query, retrieved_chunks, history)
        latency = time.perf_counter() - start_time

        self._save_turn(query, resolved_query, answer_text, retrieved_chunks)

        return RAGResponse(
            session_id=query.session_id,
            query=resolved_query.text or resolved_query.display_text(),
            answer=answer_text,
            route_taken=decision.path.value,
            confidence_score=decision.confidence,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=latency,
        )

    def answer_stream(self, query: MultiModalQuery) -> Iterator[StreamEvent]:
        """
        Synchronous generator — intended to be run on a worker thread by the
        WebSocket layer (see api/streaming_utils.py), since it performs
        blocking embedding/FAISS/BM25/LLM calls.
        """
        start_time = time.perf_counter()

        history = self._session_store.get_history(query.session_id)
        resolved_query = self._resolve_query(query, history)

        yield StreamEvent(type="status", payload={"stage": "retrieving"})
        decision, retrieved_chunks = self._route_and_retrieve(resolved_query)

        yield StreamEvent(
            type="status",
            payload={"stage": "generating", "route": decision.path.value, "confidence": decision.confidence},
        )

        answer_parts: List[str] = []
        for token in self._generator.generate_stream(resolved_query, retrieved_chunks, history):
            answer_parts.append(token)
            yield StreamEvent(type="token", payload={"text": token})

        answer_text = "".join(answer_parts).strip()
        latency = time.perf_counter() - start_time

        self._save_turn(query, resolved_query, answer_text, retrieved_chunks)

        yield StreamEvent(
            type="final",
            payload={
                "session_id": query.session_id,
                "query": resolved_query.text or resolved_query.display_text(),
                "answer": answer_text,
                "route_taken": decision.path.value,
                "confidence_score": decision.confidence,
                "latency_seconds": latency,
                "sources": [self._serialize_chunk(rc) for rc in retrieved_chunks],
            },
        )

    def index_sizes(self):
        """Exposes underlying index sizes for a lightweight health/readiness check."""
        return (
            self._light_retriever.dense_retriever_text_size(),
            self._light_retriever.dense_retriever_image_size(),
        )

    def _route_and_retrieve(self, resolved_query: MultiModalQuery):
        light_confidence = self._light_retriever.confidence(resolved_query)
        decision = self._router.decide(light_confidence)

        if decision.path == RetrievalPath.LIGHT:
            retrieved_chunks = self._light_retriever.retrieve(resolved_query, self._top_k)
        else:
            retrieved_chunks = self._heavy_retriever.retrieve(resolved_query, self._top_k)
        return decision, retrieved_chunks

    def _resolve_query(self, query: MultiModalQuery, history) -> MultiModalQuery:
        """Rewrites an elliptical follow-up's text using history; the image passes through untouched."""
        if not query.has_text or query.has_image:
            return query
        rewritten_text = self._query_contextualizer.contextualize(query.text, history)
        if rewritten_text == query.text:
            return query
        return MultiModalQuery(session_id=query.session_id, text=rewritten_text, image_path=query.image_path)
    IMAGE_TAG = "[User attached an image]"

    def _save_turn(
        self,
        original_query: MultiModalQuery,
        resolved_query: MultiModalQuery,
        answer_text: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> None:
        query_label = resolved_query.text or resolved_query.display_text()
        # Later text-only turns are answered by a text-only LLM. Tagging the saved
        # turn tells it that this answer describes an image the user shared.
        if original_query.has_image and not query_label.startswith(self.IMAGE_TAG):
            query_label = f"{self.IMAGE_TAG} {query_label}"

        self._session_store.add_turn(
            original_query.session_id,
            ConversationTurn(
                query_text=query_label,
                query_image_path=original_query.image_path,
                answer=answer_text,
                retrieved_chunks=retrieved_chunks,
            ),
        )

    @staticmethod
    def _serialize_chunk(rc: RetrievedChunk) -> dict:
        import os
        return {
            "file_name": rc.chunk.metadata.get("file_name", os.path.basename(rc.chunk.source_path)),
            "page_number": rc.chunk.page_number,
            "modality": rc.chunk.modality,
            "score": rc.score,
        }