"""
Composition root for the CLI. Offline ingestion + interactive terminal chat.
Pipeline wiring itself now lives in pipeline/pipeline_factory.py so the API
(api/dependencies.py) can reuse the exact same construction logic.
"""
import os
from typing import Optional, Tuple

import argparse

from config.settings import get_settings

from ingestion.pdf_loader import PDFDocumentLoader
from ingestion.chunker import FixedSizeTextChunker, ImageChunker
from ingestion.indexer import IndexBuilder
from retrieval.bm25_retriever import BM25CorpusStore

from pipeline.pipeline_factory import build_embedders, build_vector_stores, build_pipeline
from core.models.schemas import MultiModalQuery


def run_ingestion(settings) -> None:
    text_embedder, image_embedder = build_embedders(settings)
    text_store, image_store = build_vector_stores(settings, text_embedder.dimension, image_embedder.dimension, load_existing=False)

    document_loader = PDFDocumentLoader(image_output_dir=settings.processed_image_dir)
    text_chunker = FixedSizeTextChunker(chunk_size=settings.chunk_size_words, overlap=settings.chunk_overlap_words)
    image_chunker = ImageChunker()
    bm25_corpus_store = BM25CorpusStore(settings.bm25_corpus_path)

    index_builder = IndexBuilder(
        document_loader=document_loader,
        text_chunker=text_chunker,
        image_chunker=image_chunker,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        text_store=text_store,
        image_store=image_store,
        bm25_corpus_writer=bm25_corpus_store.save,
    )
    index_builder.build_from_folder(settings.raw_pdf_dir)

    text_store.save(settings.text_index_dir)
    image_store.save(settings.image_index_dir)
    print("Ingestion finished. Indexes saved to disk.")


def _parse_user_input(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Supported formats:
      - plain text                     -> text only
      - 'img:<path>'                   -> image only
      - '<text> | img:<path>'          -> text + image
    """
    text_part = None
    image_part = None
    for segment in (s.strip() for s in raw.split("|")):
        if not segment:
            continue
        if segment.lower().startswith("img:"):
            image_part = segment[4:].strip()
        else:
            text_part = segment if text_part is None else f"{text_part} {segment}"
    return text_part, image_part


def run_chat(settings) -> None:
    pipeline = build_pipeline(settings)
    session_id = pipeline.start_session()

    print("\n=== AdaRAG + VisRAG Hybrid Assistant ===")
    print(f"Session ID: {session_id}")
    print("Input formats:")
    print("  - plain text                      -> text-only query")
    print("  - img:/path/to/image.jpg          -> image-only query")
    print("  - some question | img:/path.jpg   -> text + image query")
    print("Commands: 'new' starts a fresh session, 'exit' quits.\n")

    while True:
        raw_line = input("You: ").strip()
        if raw_line.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if raw_line.lower() == "new":
            session_id = pipeline.start_session()
            print(f"Started new session: {session_id}\n")
            continue
        if not raw_line:
            continue

        text_part, image_part = _parse_user_input(raw_line)
        if not text_part and not image_part:
            print("Please provide text and/or an image path (see input formats above).\n")
            continue
        if image_part and not os.path.exists(image_part):
            print(f"Image path not found: {image_part}\n")
            continue

        query = MultiModalQuery(session_id=session_id, text=text_part, image_path=image_part)
        response = pipeline.answer(query)

        print(
            f"\n[Session: {response.session_id} | Route: {response.route_taken.upper()} "
            f"| Confidence: {response.confidence_score:.3f} | Latency: {response.latency_seconds:.2f}s]"
        )
        print(f"Assistant: {response.answer}\n")

        # if response.retrieved_chunks:
        #     print("Sources:")
        #     for rc in response.retrieved_chunks:
        #         label = rc.chunk.metadata.get("file_name", rc.chunk.source_path)
        #         print(f"  - {label} (page {rc.chunk.page_number}, {rc.chunk.modality}, score={rc.score:.3f})")
        print()


def main():
    parser = argparse.ArgumentParser(description="AdaRAG + VisRAG hybrid pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Build FAISS + BM25 indexes from PDFs in data/raw_pdfs")
    subparsers.add_parser("chat", help="Start an interactive terminal chat session")
    args = parser.parse_args()

    settings = get_settings()

    if args.command == "ingest":
        run_ingestion(settings)
    elif args.command == "chat":
        run_chat(settings)


if __name__ == "__main__":
    main()