"""
Optional Redis-backed task queue for PDF processing.

By default the application keeps relying on the in-process
`DatabaseWatcher` (polls Mongo for `status="uploaded"` rows). Set
`ENABLE_PDF_WORKER=true` in the environment and run a separate
`arq chatbot.workers.pdf_worker.WorkerSettings` process to switch to
push-based processing:

  * `/chat/upload` enqueues a job after saving the file to GridFS,
  * the arq worker picks it up, claims the row, runs the heavy
    GenAI vectorisation, and sets `status="processed"`.

The watcher is left intact as a self-healing safety net for rows
inserted before the worker was up (e.g. CLI uploads, or jobs lost
due to a Redis flush). Enabling the worker just changes WHO does
the work first.

This module is intentionally light on imports — `arq` is only
imported lazily inside the helpers so importing it from contexts
that don't need the queue (tests, migration scripts) doesn't pull
the arq dependency.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from chatbot.config import config as app_config

logger = logging.getLogger(__name__)


# --- Feature flag -----------------------------------------------------

def is_worker_enabled() -> bool:
    """True when ENABLE_PDF_WORKER is set to a truthy value AND Redis URL exists."""
    flag = os.getenv("ENABLE_PDF_WORKER", "false").lower() in ("1", "true", "yes", "on")
    return flag and bool(app_config.REDIS_URL)


# --- Queue / job naming (kept as constants so worker + API agree) ------

PDF_QUEUE_NAME = "lexmind:pdf"
PROCESS_PDF_JOB = "process_pdf"


# --- Redis settings ----------------------------------------------------

def _redis_settings():
    """Build `arq.connections.RedisSettings` from `REDIS_URL`.

    Imported lazily so this module remains importable without arq
    installed (the worker extras pull it in).
    """
    if not app_config.REDIS_URL:
        raise RuntimeError(
            "REDIS_URL is not configured; cannot use the PDF worker queue."
        )
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(app_config.REDIS_URL)


# --- Enqueue helper ----------------------------------------------------

async def enqueue_pdf_processing(doc_id: str) -> Optional[str]:
    """
    Enqueue a job to vectorise the document identified by `doc_id`.

    Returns the arq job id on success, or None when the worker is
    disabled / Redis is unreachable. Failure is non-fatal — the
    in-process watcher will still pick the row up on its next poll,
    so the upload endpoint should not return an error to the user.
    """
    if not is_worker_enabled():
        logger.debug("[queue] PDF worker disabled; relying on watcher fallback")
        return None

    try:
        from arq import create_pool
    except ImportError:
        logger.warning(
            "[queue] arq is not installed; skipping enqueue. "
            "Install with `pip install arq` or set ENABLE_PDF_WORKER=false."
        )
        return None

    try:
        pool = await create_pool(_redis_settings())
    except Exception as exc:
        logger.warning("[queue] could not connect to Redis: %s", exc)
        return None

    try:
        job = await pool.enqueue_job(
            PROCESS_PDF_JOB,
            doc_id,
            _queue_name=PDF_QUEUE_NAME,
        )
        if job is None:
            logger.debug("[queue] job %s already enqueued (deduped)", doc_id)
            return None
        return job.job_id
    except Exception as exc:
        logger.warning("[queue] enqueue failed for doc_id=%s: %s", doc_id, exc)
        return None
    finally:
        try:
            await pool.close()
        except Exception:
            pass
