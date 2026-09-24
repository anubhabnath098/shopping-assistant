"""
Pydantic response contracts for the API layer. REST requests are plain
multipart form fields (see routes.py); WebSocket messages are plain JSON,
documented here for reference and for optional validation in websocket_routes.py.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatSourceItem(BaseModel):
    file_name: str
    page_number: int
    modality: str  # "text" | "image"
    score: float


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Echo back on the next call to continue this conversation.")
    query: str
    answer: str
    route_taken: str
    confidence_score: float
    latency_seconds: float
    sources: List[ChatSourceItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    index_text_chunks: int
    index_image_chunks: int


# --- WebSocket message contracts (client -> server) ---

class WSIncomingMessage(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Omit on first message to start a new session.")
    text: Optional[str] = Field(default=None, description="Transcribed speech text, if any.")
    image_base64: Optional[str] = Field(default=None, description="Base64/data-URL of a captured camera frame, if any.")


# --- WebSocket message contracts (server -> client), for reference only ---
# {"type": "session", "session_id": "..."}                                  -> sent once, first ever reply
# {"type": "status", "stage": "retrieving" | "generating", ...}             -> progress updates
# {"type": "token", "text": "..."}                                         -> incremental answer text
# {"type": "final", "session_id", "query", "answer", "route_taken",
#  "confidence_score", "latency_seconds", "sources": [ChatSourceItem, ...]} -> end of turn
# {"type": "error", "message": "..."}                                      -> something went wrong