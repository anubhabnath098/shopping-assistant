"""
FastAPI application factory. Run via `python run_api.py` or
`uvicorn api.app:app --reload` during frontend development.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from api.routes import router
from api.websocket_routes import ws_router
from api.dependencies import get_pipeline


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AdaRAG + VisRAG Hybrid API",
        description="Multimodal, session-aware RAG backend (text / image / text+image queries, REST + WebSocket).",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")
    app.include_router(ws_router, prefix="/api")  # -> ws://host:port/api/ws/chat

    @app.on_event("startup")
    def _warm_up_pipeline() -> None:
        # Build (and cache) the pipeline at startup rather than on the first
        # incoming request/connection, so the first user doesn't pay the
        # model-load cost.
        get_pipeline()

    return app


app = create_app()