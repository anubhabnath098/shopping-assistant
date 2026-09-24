"""
Real-time, session-aware, multimodal chat over a persistent WebSocket
connection. One connection typically maps to one ongoing conversation
(voice-driven client keeps it open for the whole session).

Client -> Server message (JSON):
    {"session_id": "abc" | null, "text": "..." | null, "image_base64": "..." | null}

Server -> Client messages (JSON), in order per query:
    {"type": "session", "session_id": "..."}         (only if a new session was created)
    {"type": "status", "stage": "retrieving"}
    {"type": "status", "stage": "generating", "route": "light"|"heavy", "confidence": 0.71}
    {"type": "token", "text": "..."}                 (repeated, as the LLM streams)
    {"type": "final", ...full answer + sources...}
    {"type": "error", "message": "..."}              (on failure)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.settings import get_settings
from core.models.schemas import MultiModalQuery
from pipeline.rag_pipeline import RAGPipeline
from api.dependencies import get_pipeline
from api.image_utils import save_base64_image
from api.streaming_utils import stream_pipeline_answer

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    pipeline: RAGPipeline = get_pipeline()
    settings = get_settings()
    active_session_id: str | None = None

    try:
        while True:
            raw_message = await websocket.receive_json()

            client_session_id = raw_message.get("session_id") or active_session_id
            text = raw_message.get("text")
            image_base64 = raw_message.get("image_base64")

            if not (text and str(text).strip()) and not image_base64:
                await websocket.send_json({"type": "error", "message": "Provide 'text' and/or 'image_base64'."})
                continue

            resolved_session_id = pipeline.ensure_session(client_session_id)
            is_new_session = resolved_session_id != client_session_id
            active_session_id = resolved_session_id

            if is_new_session:
                await websocket.send_json({"type": "session", "session_id": resolved_session_id})

            image_path = None
            if image_base64:
                try:
                    image_path = save_base64_image(image_base64, settings.api_upload_dir)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

            query = MultiModalQuery(
                session_id=resolved_session_id,
                text=str(text).strip() if text else None,
                image_path=image_path,
            )

            async for event in stream_pipeline_answer(pipeline, query):
                await websocket.send_json({"type": event.type, **event.payload})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass