"""
PDF document lifecycle:
  1. `save_pdf_to_mongo`  — store the binary in GridFS + metadata row
  2. `process_and_vectorize_pdf` — parse with opendataloader-pdf,
     chunk by legal structure, embed via Cohere, upsert into the
     `user_uploaded` Qdrant collection
  3. `get_session_doc_ids` — list processed doc ids for a session
     (used by `tool_search_uploaded_file` to filter Qdrant search)

The previous version delegated parsing + retrieval to Google's managed
`file_search_stores`. We now own the full pipeline; see the PR
description for the migration story.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from bson.objectid import ObjectId

from chatbot.core.chunker import chunk_legal_document
from chatbot.core.db import DB_DOCUMENTS_COLLECTION, FS
from chatbot.core.utils import compute_file_hash

logger = logging.getLogger(__name__)


def save_pdf_to_mongo(
    file_path: str,
    session_id: str,
    user_id: str,
    original_filename: str | None = None,
) -> str | None:
    """Save PDF into GridFS + documents collection. Return `_id` as string."""
    fs_client = FS
    coll = DB_DOCUMENTS_COLLECTION
    if fs_client is None or coll is None:
        logger.warning("[file_store] DB or FS not ready")
        return None
    try:
        file_hash = compute_file_hash(file_path)
        file_name = original_filename or os.path.basename(file_path)

        # Dedup: same hash in same session
        existing = coll.find_one(
            {"file_hash": file_hash, "user_id": user_id, "session_id": session_id}
        )
        if existing:
            return str(existing["_id"])

        # Reuse GridFS blob if the user already uploaded this file before
        hash_existing = coll.find_one({"file_hash": file_hash, "user_id": user_id})
        if hash_existing:
            file_gridfs_id = hash_existing["file_gridfs_id"]
        else:
            with open(file_path, "rb") as f:
                gridfs_id = fs_client.put(
                    f, filename=file_name, metadata={"original_user": user_id}
                )
            file_gridfs_id = str(gridfs_id)

        result = coll.insert_one(
            {
                "user_id": user_id,
                "session_id": session_id,
                "filename": file_name,
                "file_gridfs_id": file_gridfs_id,
                "file_hash": file_hash,
                "created_at": datetime.now(timezone.utc),
                "status": "uploaded",
            }
        )
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("[file_store.save_pdf_to_mongo] %s", exc)
        return None


def process_and_vectorize_pdf(
    file_path: str,
    session_id: str,
    doc_id: str,
    *,
    embedder,
    vector_store,
    user_id: Optional[str] = None,
) -> bool:
    """Parse → chunk → embed → upsert into the user-upload collection.

    On success flips the Mongo row to `status=processed` with
    `chunk_count` and `processed_at`. On failure flips to
    `status=error_processing` with the exception message — same
    contract as the previous Google-managed implementation, so the
    `/chat/files` and watcher status reporting keeps working.

    Returns True on success.
    """
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        logger.warning("[file_store] DB not ready")
        return False
    if embedder is None or vector_store is None:
        logger.warning("[file_store] embedder/vector_store not provided")
        return False

    file_name = os.path.basename(file_path)
    try:
        chunks = _parse_and_chunk(file_path, source_file=file_name, doc_id=doc_id)
        if not chunks:
            raise RuntimeError("opendataloader-pdf produced no chunks")

        embeddings = embedder.embed_documents([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks"
            )

        # Build payloads and upsert
        from chatbot.core.vectorstore import chunks_to_points

        ids, vecs, payloads = chunks_to_points(
            chunks,
            embeddings,
            extra_payload={"session_id": session_id, "user_id": user_id} if session_id else {},
        )
        vector_store.upsert(
            collection=vector_store.user_collection,
            ids=ids,
            embeddings=vecs,
            payloads=payloads,
        )

        coll.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "status": "processed",
                    "chunk_count": len(chunks),
                    "processed_at": datetime.now(timezone.utc),
                }
            },
        )
        logger.info(
            "[file_store] processed %s — %d chunks indexed", file_name, len(chunks)
        )
        return True

    except Exception as exc:
        logger.error("[file_store.process_and_vectorize_pdf] %s", exc)
        try:
            coll.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"status": "error_processing", "error": str(exc)}},
            )
        except Exception:
            pass
        return False


def _parse_and_chunk(file_path: str, *, source_file: str, doc_id: str):
    """Run opendataloader-pdf into a tempdir and chunk the resulting JSON.

    Imported lazily so tests / migrations that don't need PDF parsing
    don't pay for the Java launch on module import.
    """
    import json

    import opendataloader_pdf

    with tempfile.TemporaryDirectory() as out_dir:
        opendataloader_pdf.convert(
            file_path,
            output_dir=out_dir,
            format="json",
            quiet=True,
        )
        # opendataloader writes `<input_stem>.json`
        json_files = [f for f in os.listdir(out_dir) if f.endswith(".json")]
        if not json_files:
            raise RuntimeError("opendataloader-pdf produced no JSON output")
        json_path = os.path.join(out_dir, json_files[0])
        with open(json_path, "r", encoding="utf-8") as f:
            doc_json = json.load(f)

    return chunk_legal_document(
        doc_json,
        source_file=source_file,
        doc_id=doc_id,
    )


def get_session_doc_ids(session_id: str) -> list[str]:
    """Return list of processed `doc_id`s for a session.

    Used by `tool_search_uploaded_file` to filter the Qdrant query by
    `doc_id IN (...)` so we only retrieve chunks from THIS session's
    files even though all users' uploads share one collection.
    """
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return []
    try:
        cursor = coll.find(
            {"session_id": session_id, "status": "processed"}, {"_id": 1}
        )
        return [str(doc["_id"]) for doc in cursor]
    except Exception:
        return []
