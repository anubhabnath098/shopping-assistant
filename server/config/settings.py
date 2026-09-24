"""
Single source of truth for all configuration. Change a model name, a
threshold, or a path here — nothing else needs touching.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    # --- Paths ---
    base_dir: str = BASE_DIR
    raw_pdf_dir: str = os.path.join(BASE_DIR, "data", "raw_pdfs")
    processed_image_dir: str = os.path.join(BASE_DIR, "data", "processed", "images")
    text_index_dir: str = os.path.join(BASE_DIR, "data", "indexes", "text_index")
    image_index_dir: str = os.path.join(BASE_DIR, "data", "indexes", "image_index")
    bm25_corpus_path: str = os.path.join(BASE_DIR, "data", "indexes", "bm25_corpus.pkl")
    api_upload_dir: str = os.path.join(BASE_DIR, "data", "uploads")

    # --- Embedding models (swap via factory registry, not by editing pipeline code) ---
    text_embedder_type: str = "minilm"
    image_embedder_type: str = "clip"

    # --- Vector store ---
    vector_store_type: str = "faiss"
    vector_index_type: str = "hnsw"  # fast ANN, used for both light & heavy paths

    # --- LLM ---
    llm_type: str = "gemini"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 1024

    # --- Chunking ---
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40

    # --- Retrieval / AdaRAG routing ---
    top_k: int = 5
    heavy_pool_size: int = 20
    adarag_confidence_threshold: float = 0.55
    text_modality_weight: float = 0.6
    image_modality_weight: float = 0.4

    # --- Session / conversational memory ---
    max_history_turns: int = 6

    # --- API server ---
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_cors_allow_origins: tuple = ("*",)  # tighten for production frontends


_settings_instance: "Settings | None" = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance