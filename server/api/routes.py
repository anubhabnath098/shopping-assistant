"""
HTTP surface of the RAG system. Deliberately thin — all real logic lives in
the pipeline/retrieval/generation layers; this module only translates
HTTP <-> domain objects.
"""
import os
from typing import Optional

from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException

from config.settings import get_settings
from core.models.schemas import MultiModalQuery
from pipeline.rag_pipeline import RAGPipeline
from api.dependencies import get_pipeline
from api.schemas import ChatResponse, ChatSourceItem, HealthResponse
from api.image_utils import save_uploaded_file

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    text_size, image_size = pipeline.index_sizes()
    return HealthResponse(status="ok", index_text_chunks=text_size, index_image_chunks=image_size)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    session_id: Optional[str] = Form(default=None, description="Omit to start a new conversation."),
    text: Optional[str] = Form(default=None, description="The user's text query, if any."),
    image: Optional[UploadFile] = File(default=None, description="The user's image query, if any."),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ChatResponse:
    if not (text and text.strip()) and image is None:
        raise HTTPException(status_code=400, detail="Provide at least one of 'text' or 'image'.")

    resolved_session_id = pipeline.ensure_session(session_id)

    saved_image_path = None
    if image is not None:
        try:
            saved_image_path = await save_uploaded_file(image, get_settings().api_upload_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    query = MultiModalQuery(
        session_id=resolved_session_id,
        text=text.strip() if text else None,
        image_path=saved_image_path,
    )
    response = pipeline.answer(query)

    return ChatResponse(
        session_id=response.session_id,
        query=response.query,
        answer=response.answer,
        route_taken=response.route_taken,
        confidence_score=response.confidence_score,
        latency_seconds=response.latency_seconds,
        sources=[
            ChatSourceItem(
                file_name=rc.chunk.metadata.get("file_name", os.path.basename(rc.chunk.source_path)),
                page_number=rc.chunk.page_number,
                modality=rc.chunk.modality,
                score=rc.score,
            )
            for rc in response.retrieved_chunks
        ],
    )