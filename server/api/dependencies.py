"""
FastAPI dependency providers. The pipeline is expensive to build (loads
embedding models + FAISS indexes), so it is constructed exactly once per
process and reused across all requests.
"""
from functools import lru_cache

from config.settings import get_settings
from pipeline.pipeline_factory import build_pipeline
from pipeline.rag_pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    return build_pipeline(settings)