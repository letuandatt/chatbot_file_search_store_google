"""
arq worker for PDF vectorisation.

Run with:

    arq chatbot.workers.pdf_worker.WorkerSettings

Set `ENABLE_PDF_WORKER=true` so `/chat/upload` enqueues jobs into
this worker; otherwise files only flow through the in-process
`DatabaseWatcher` poller.

The worker uses the same `DatabaseWatcher._process_single_file`
code path as the watcher (kept as a method on the class) — there is
only ONE place where GenAI upload + Mongo status flip happens, so
the worker and the watcher cannot drift apart.
"""
from __future__ import annotations

import logging
from typing import Any

from bson.objectid import ObjectId
from pymongo import ReturnDocument

from chatbot.core.db import DB_DOCUMENTS_COLLECTION, init_db
from chatbot.core.queue import PDF_QUEUE_NAME, _redis_settings
from chatbot.core.watcher import app_watcher

logger = logging.getLogger(__name__)


async def process_pdf(ctx: dict[str, Any], doc_id: str) -> str:
    """
    Process one uploaded PDF identified by Mongo `_id`.

    The function:
      1. Atomically claims the row (status → 'processing'); if the
         row was already claimed (e.g. by the watcher) we exit
         silently — that's the deduplication that lets the worker
         and the watcher coexist safely.
      2. Delegates the heavy lifting to
         `DatabaseWatcher._process_single_file`, which handles
         GridFS read, Google upload, and the final status flip.
    """
    if DB_DOCUMENTS_COLLECTION is None:
        init_db()

    try:
        oid = ObjectId(doc_id)
    except Exception:
        logger.warning("[worker] invalid doc_id=%s; dropping job", doc_id)
        return "invalid_doc_id"

    claimed = DB_DOCUMENTS_COLLECTION.find_one_and_update(
        {"_id": oid, "status": "uploaded"},
        {"$set": {"status": "processing"}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        # Someone else (the watcher, or a previous worker run) already
        # picked this up. That's fine — exit cleanly so arq marks the
        # job complete without retrying.
        logger.info("[worker] doc_id=%s already claimed; skipping", doc_id)
        return "already_claimed"

    # The watcher's per-file processing is the single source of truth
    # for GenAI upload + status flip; we deliberately reuse it instead
    # of duplicating the logic in the worker.
    app_watcher._process_single_file(claimed)
    return "processed"


async def startup(ctx: dict[str, Any]) -> None:
    init_db()
    logger.info("[worker] PDF worker ready")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("[worker] PDF worker shutting down")


class _LazyRedisSettings:
    """Descriptor that defers `RedisSettings.from_dsn(REDIS_URL)` until
    arq actually reads `WorkerSettings.redis_settings`. This lets the
    module be imported in environments without REDIS_URL (e.g. unit
    tests, the API process when the worker is disabled)."""

    def __get__(self, _instance, _owner):
        return _redis_settings()


class WorkerSettings:
    """arq worker entry point. See module docstring."""

    functions = [process_pdf]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = PDF_QUEUE_NAME
    redis_settings = _LazyRedisSettings()
    max_jobs = 4
    job_timeout = 600  # 10 minutes; GenAI upload + vectorisation can be slow
    keep_result = 300  # seconds
