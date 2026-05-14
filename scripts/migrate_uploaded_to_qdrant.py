"""
One-shot migration: re-process every "processed" user-uploaded PDF
through the new opendataloader + Cohere + Qdrant pipeline.

Why this exists:

The old ingestion path stored the document text inside Google's
managed `file_search_stores` — there is no programmatic way to pull
it back out. Each Mongo row carried only a `file_store_name`
reference. After PR R lands, those refs point at a place we no longer
query, so old uploads become orphaned in the UI.

This script walks the Mongo `documents` collection, rebuilds the
chunks from the original PDF (still in GridFS), embeds them, and
upserts into the `user_uploaded` Qdrant collection. Idempotent —
chunk ids are deterministic so re-running is safe.

Usage:

    # After QDRANT is up and `.env` has QDRANT_URL set:
    python -m scripts.migrate_uploaded_to_qdrant
    python -m scripts.migrate_uploaded_to_qdrant --user-id <uid>
    python -m scripts.migrate_uploaded_to_qdrant --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

from bson.objectid import ObjectId

from chatbot.core.db import DB_DOCUMENTS_COLLECTION, FS, init_db
from chatbot.core.embedder import CohereEmbedder
from chatbot.core.file_store import process_and_vectorize_pdf
from chatbot.core.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


def migrate(
    *, user_id: str | None = None, only_unmigrated: bool = True, dry_run: bool = False
) -> dict:
    init_db()
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None or FS is None:
        raise RuntimeError("DB/GridFS not available")

    embedder = CohereEmbedder()
    vector_store = QdrantVectorStore()
    vector_store.ensure_collections()

    # We migrate every doc whose binary is still in GridFS, regardless
    # of its status — the new pipeline owns the `status` field after
    # this point.
    q: dict = {"file_gridfs_id": {"$exists": True}}
    if user_id:
        q["user_id"] = user_id
    if only_unmigrated:
        # Skip rows already touched by the new pipeline.
        q["chunk_count"] = {"$exists": False}

    stats = {"total": 0, "migrated": 0, "failed": 0}

    cursor = coll.find(q)
    for doc in cursor:
        stats["total"] += 1
        filename = doc.get("filename", "unknown.pdf")
        gridfs_id = doc.get("file_gridfs_id")

        logger.info("[%d] %s", stats["total"], filename)

        temp_path = None
        try:
            grid_out = FS.get(ObjectId(gridfs_id))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(grid_out.read())
                temp_path = tmp.name

            if dry_run:
                logger.info("  dry-run: would migrate %s", filename)
                continue

            ok = process_and_vectorize_pdf(
                file_path=temp_path,
                session_id=doc.get("session_id", ""),
                doc_id=str(doc["_id"]),
                embedder=embedder,
                vector_store=vector_store,
                user_id=doc.get("user_id"),
            )
            if ok:
                stats["migrated"] += 1
            else:
                stats["failed"] += 1
        except Exception as exc:
            logger.error("  migration failed: %s", exc)
            stats["failed"] += 1
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="Migrate uploads belonging to one user only")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force re-migrating even docs that already have chunk_count (re-embed)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = migrate(
        user_id=args.user_id,
        only_unmigrated=not args.all,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 50)
    print("USER-UPLOAD MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Documents scanned:  {stats['total']}")
    print(f"Migrated:           {stats['migrated']}")
    print(f"Failed:             {stats['failed']}")
    print("=" * 50)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
