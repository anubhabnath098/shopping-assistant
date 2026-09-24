"""
Bridges RAGPipeline.answer_stream() — a synchronous, blocking generator
(embedding calls, FAISS search, Gemini streaming) — onto the asyncio event
loop used by the WebSocket handler. Without this, a single slow query would
block FastAPI's event loop and stall every other connected client.
"""
import asyncio
import queue
import threading
from typing import AsyncIterator

from core.models.schemas import StreamEvent, MultiModalQuery
from pipeline.rag_pipeline import RAGPipeline

_SENTINEL = object()


async def stream_pipeline_answer(pipeline: RAGPipeline, query: MultiModalQuery) -> AsyncIterator[StreamEvent]:
    loop = asyncio.get_running_loop()
    thread_queue: "queue.Queue" = queue.Queue()

    def _run_in_background_thread() -> None:
        try:
            for event in pipeline.answer_stream(query):
                thread_queue.put(event)
        except Exception as exc:  # surface any unexpected failure to the client instead of hanging it
            thread_queue.put(StreamEvent(type="error", payload={"message": str(exc)}))
        finally:
            thread_queue.put(_SENTINEL)

    worker = threading.Thread(target=_run_in_background_thread, daemon=True)
    worker.start()

    while True:
        event = await loop.run_in_executor(None, thread_queue.get)
        if event is _SENTINEL:
            break
        yield event