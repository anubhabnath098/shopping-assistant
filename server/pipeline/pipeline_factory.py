"""
Single place that knows how to assemble a fully-wired RAGPipeline from
Settings. Both the CLI (main.py) and the API (api/dependencies.py) call
into this factory instead of duplicating the wiring logic.
"""
import os

from config.settings import Settings

from factories.embedder_factory import TextEmbedderFactory, ImageEmbedderFactory
from factories.vector_store_factory import VectorStoreFactory
from factories.llm_factory import LLMFactory

from retrieval.bm25_retriever import BM25Retriever, BM25CorpusStore
from retrieval.dual_index_retriever import DualIndexRetriever
from retrieval.light_retriever import LightRetriever
from retrieval.heavy_retriever import HeavyRetriever
from retrieval.reranker import ReciprocalRankFusionReranker
from retrieval.router import AdaptiveRouter

from generation.prompt_builder import PromptBuilder
from generation.generator import ResponseGenerator

from sessions.in_memory_session_store import InMemorySessionStore
from sessions.query_contextualizer import QueryContextualizer

from pipeline.rag_pipeline import RAGPipeline


def build_embedders(settings: Settings):
    text_embedder = TextEmbedderFactory.create(settings.text_embedder_type)
    image_embedder = ImageEmbedderFactory.create(settings.image_embedder_type)
    return text_embedder, image_embedder


def build_vector_stores(settings: Settings, text_dim: int, image_dim: int, load_existing: bool):
    text_store = VectorStoreFactory.create(settings.vector_store_type, dimension=text_dim, index_type=settings.vector_index_type)
    image_store = VectorStoreFactory.create(settings.vector_store_type, dimension=image_dim, index_type=settings.vector_index_type)
    if load_existing:
        if os.path.exists(settings.text_index_dir):
            text_store.load(settings.text_index_dir)
        if os.path.exists(settings.image_index_dir):
            image_store.load(settings.image_index_dir)
    return text_store, image_store


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Assembles the full AdaRAG + VisRAG pipeline, ready to answer queries."""
    text_embedder, image_embedder = build_embedders(settings)
    text_store, image_store = build_vector_stores(
        settings, text_embedder.dimension, image_embedder.dimension, load_existing=True
    )

    dense_retriever = DualIndexRetriever(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        text_store=text_store,
        image_store=image_store,
        text_weight=settings.text_modality_weight,
        image_weight=settings.image_modality_weight,
    )
    light_retriever = LightRetriever(dense_retriever)

    bm25_corpus_store = BM25CorpusStore(settings.bm25_corpus_path)
    bm25_retriever = BM25Retriever(bm25_corpus_store)
    reranker = ReciprocalRankFusionReranker()
    heavy_retriever = HeavyRetriever(bm25_retriever, dense_retriever, reranker, pool_size=settings.heavy_pool_size)

    router = AdaptiveRouter(confidence_threshold=settings.adarag_confidence_threshold)

    text_llm_provider = LLMFactory.create(
        settings.text_llm_type,              # e.g. "qwen"
        api_key=settings.text_llm_api_key,
        model_name=settings.text_llm_model_name,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    vision_llm_provider = LLMFactory.create(
        settings.vision_llm_type,            # e.g. "gemini"
        api_key=settings.vision_llm_api_key,
        model_name=settings.vision_llm_model_name,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    generator = ResponseGenerator(text_llm_provider, vision_llm_provider, PromptBuilder())

    session_store = InMemorySessionStore(max_turns=settings.max_history_turns)
    # Follow-up rewriting is always text-only, even when the original turn had
    # an image (see QueryContextualizer.contextualize's `has_text and not has_image` gate
    # in _resolve_query), so it should always use the text model.
    query_contextualizer = QueryContextualizer(text_llm_provider)

    return RAGPipeline(
        light_retriever=light_retriever,
        heavy_retriever=heavy_retriever,
        router=router,
        generator=generator,
        session_store=session_store,
        query_contextualizer=query_contextualizer,
        top_k=settings.top_k,
    )