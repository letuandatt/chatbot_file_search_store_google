"""
Offline ingestion script for the LAW corpus.

Replaces the previous Google `file_search_stores` upload flow. The new
pipeline is fully self-hosted:

    for each PDF in DATA_DIR:
        opendataloader_pdf.convert(..., format="json")
            ↓  produces a per-PDF JSON tree
        chunk_legal_document(tree)
            ↓  groups blocks by `Điều X` (legal-aware chunking)
        embedder.embed_documents(chunks)
            ↓  Cohere embed-multilingual-v3 → 1024-d vectors
        vector_store.upsert(law_collection, ...)
            ↓  Qdrant

Usage (after `pip install opendataloader-pdf qdrant-client` and
booting Qdrant via `docker compose up qdrant`):

    python -m chatbot.setup_main_store.setup_main_store

    python -m chatbot.setup_main_store.setup_main_store --data-dir /path/to/pdfs
    python -m chatbot.setup_main_store.setup_main_store --dry-run

The script is **idempotent**: chunk ids are deterministic from
`(source_file, section, page, text-prefix)` so re-running upserts in
place rather than duplicating rows. Safe to interrupt and resume — the
work unit is one PDF at a time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from chatbot.config import config as app_config
from chatbot.core.chunker import LegalChunk, chunk_legal_document
from chatbot.core.embedder import CohereEmbedder
from chatbot.core.vectorstore import QdrantVectorStore, chunks_to_points

logger = logging.getLogger(__name__)


def iter_pdf_paths(root: Path) -> Iterable[Path]:
    """Yield every `*.pdf` under *root* (recursive), sorted for determinism."""
    yield from sorted(root.rglob("*.pdf"))


def parse_pdf_to_chunks(pdf_path: Path, *, output_dir: Path) -> list[LegalChunk]:
    """Run opendataloader-pdf and chunk the resulting JSON."""
    import opendataloader_pdf

    opendataloader_pdf.convert(
        str(pdf_path),
        output_dir=str(output_dir),
        format="json",
        quiet=True,
    )

    stem = pdf_path.stem
    json_path = output_dir / f"{stem}.json"
    if not json_path.exists():
        # Some PDFs may produce a different filename if the input
        # contained unusual characters; fall back to the first json
        # file produced.
        candidates = list(output_dir.glob("*.json"))
        if not candidates:
            raise RuntimeError(f"opendataloader produced no JSON for {pdf_path.name}")
        json_path = candidates[0]

    with open(json_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    chunks = chunk_legal_document(
        doc_json,
        source_file=pdf_path.name,
        # For the law corpus we don't have a Mongo doc id; we seed
        # chunk ids from the filename only, which is stable because
        # PDF filenames are the canonical identifier for these docs.
        doc_id=pdf_path.stem,
    )
    return chunks


def ingest_corpus(
    data_dir: Path,
    *,
    embedder: CohereEmbedder,
    vector_store: QdrantVectorStore,
    dry_run: bool = False,
    batch_size: int = 32,
) -> dict:
    """Walk `data_dir`, ingest every PDF.

    Returns counters useful for the CLI summary printout.
    """
    vector_store.ensure_collections()
    collection = vector_store.law_collection

    stats = {"pdfs": 0, "chunks": 0, "embedded": 0, "upserted": 0, "skipped": 0}
    pdf_paths = list(iter_pdf_paths(data_dir))
    if not pdf_paths:
        logger.warning("No PDFs found under %s", data_dir)
        return stats

    logger.info("Found %d PDFs under %s", len(pdf_paths), data_dir)

    for pdf_path in pdf_paths:
        stats["pdfs"] += 1
        logger.info("[%d/%d] %s", stats["pdfs"], len(pdf_paths), pdf_path.name)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                chunks = parse_pdf_to_chunks(pdf_path, output_dir=Path(out_dir))
        except Exception as exc:
            logger.error("  parse failed: %s", exc)
            stats["skipped"] += 1
            continue

        if not chunks:
            logger.warning("  no chunks extracted; skipping")
            stats["skipped"] += 1
            continue

        stats["chunks"] += len(chunks)
        logger.info("  parsed %d chunks", len(chunks))

        if dry_run:
            continue

        # Embed and upsert in batches so a single PDF doesn't try to
        # blast 5000 chunks through Cohere in one shot.
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            try:
                vectors = embedder.embed_documents([c.text for c in batch])
            except Exception as exc:
                logger.error("  embed batch failed: %s", exc)
                continue
            stats["embedded"] += len(vectors)

            ids, vecs, payloads = chunks_to_points(batch, vectors)
            try:
                vector_store.upsert(
                    collection=collection,
                    ids=ids,
                    embeddings=vecs,
                    payloads=payloads,
                )
                stats["upserted"] += len(ids)
            except Exception as exc:
                logger.error("  upsert batch failed: %s", exc)

        # Be polite to Cohere's rate limiter between PDFs.
        time.sleep(0.2)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(app_config.DATA_DIR),
        help="Directory containing law PDFs (recursive). Default: $DATA_DIR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; skip Cohere embed and Qdrant upsert.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Chunks per embed/upsert batch.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Python logging level.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data dir not found: %s", data_dir)
        return 2

    embedder = CohereEmbedder()
    vector_store = QdrantVectorStore()

    stats = ingest_corpus(
        data_dir=data_dir,
        embedder=embedder,
        vector_store=vector_store,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    print()
    print("=" * 50)
    print("LAW CORPUS INGESTION COMPLETE")
    print("=" * 50)
    print(f"PDFs scanned:    {stats['pdfs']}")
    print(f"Chunks parsed:   {stats['chunks']}")
    print(f"Chunks embedded: {stats['embedded']}")
    print(f"Chunks upserted: {stats['upserted']}")
    print(f"PDFs skipped:    {stats['skipped']}")
    print(f"Collection:      {vector_store.law_collection}")
    print(f"Qdrant URL:      {app_config.QDRANT_URL}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
