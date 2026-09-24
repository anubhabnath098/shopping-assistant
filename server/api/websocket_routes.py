"""
Real-time, session-aware, multimodal chat over a persistent WebSocket
connection. One connection typically maps to one ongoing conversation
(voice-driven client keeps it open for the whole session).

Queries run CONCURRENTLY: an image query (slow, vision LLM) does not block
text queries sent afterwards on the same connection/session.

Client -> Server message (JSON):
    {"request_id": "r1" | null, "session_id": "abc" | null,
     "text": "..." | null, "image_base64": "..." | null}

Server -> Client messages (JSON). Every per-query message carries:
    "request_id": echoes the client's id (or a server-generated one)
    "kind": "image" if the query carried an image, else "text"

    {"type": "session", "session_id": "..."}          (only if a new session was created)
    {"type": "accepted", "request_id", "kind", "session_id"}
    {"type": "status", "request_id", "kind", "stage": "retrieving"}
    {"type": "status", "request_id", "kind", "stage": "generating", "route": ..., "confidence": ...}
    {"type": "token", "request_id", "kind", "text": "..."}   (text queries only)
    {"type": "final", "request_id", "kind", ...full answer + sources...}
    {"type": "error", "request_id", "kind", "message": "..."}

Because responses can arrive out of order, the client must route by request_id.
"""
import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import get_settings
from core.models.schemas import MultiModalQuery
from pipeline.rag_pipeline import RAGPipeline
from api.dependencies import get_pipeline
from api.image_utils import save_base64_image
from api.streaming_utils import stream_pipeline_answer

ws_router = APIRouter()

# Image answers come from a slower vision model; deliver them whole in the
# "final" event instead of token-by-token. Set to False to stream them too.
SUPPRESS_IMAGE_TOKENS = True


@ws_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    pipeline: RAGPipeline = get_pipeline()
    settings = get_settings()
    active_session_id: Optional[str] = None

    # Several tasks send on the same socket, so serialize the writes.
    send_lock = asyncio.Lock()
    tasks: set = set()

    async def send(event: dict) -> None:
        async with send_lock:
            await websocket.send_json(event)

    async def run_query(request_id: str, kind: str, query: MultiModalQuery) -> None:
        try:
            await send({
                "type": "accepted",
                "request_id": request_id,
                "kind": kind,
                "session_id": query.session_id,
            })
            async for event in stream_pipeline_answer(pipeline, query):
                if kind == "image" and SUPPRESS_IMAGE_TOKENS and event.type == "token":
                    continue
                await send({**event.payload, "type": event.type, "request_id": request_id, "kind": kind})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                await send({"type": "error", "request_id": request_id, "kind": kind, "message": str(exc)})
            except Exception:
                pass

    try:
        while True:
            raw_message = await websocket.receive_json()

            request_id = str(raw_message.get("request_id") or uuid.uuid4().hex)
            client_session_id = (
                raw_message["session_id"] if "session_id" in raw_message else active_session_id
            )
            text = raw_message.get("text")
            image_base64 = raw_message.get("image_base64")
            kind = "image" if image_base64 else "text"

            if not (text and str(text).strip()) and not image_base64:
                await send({
                    "type": "error", "request_id": request_id, "kind": kind,
                    "message": "Provide 'text' and/or 'image_base64'.",
                })
                continue

            # Session resolution stays in the receive loop (not in the task) so
            # that two quick messages can never create two different sessions.
            resolved_session_id = pipeline.ensure_session(client_session_id)
            is_new_session = resolved_session_id != client_session_id
            active_session_id = resolved_session_id

            if is_new_session:
                await send({"type": "session", "session_id": resolved_session_id})

            image_path = None
            if image_base64:
                try:
                    image_path = save_base64_image(image_base64, settings.api_upload_dir)
                except ValueError as exc:
                    await send({"type": "error", "request_id": request_id, "kind": kind, "message": str(exc)})
                    continue

            query = MultiModalQuery(
                session_id=resolved_session_id,
                text=str(text).strip() if text else None,
                image_path=image_path,
            )

            # Run in the background so the loop can immediately receive the next message.
            task = asyncio.create_task(run_query(request_id, kind, query))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await send({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        for t in tasks:
            t.cancel()