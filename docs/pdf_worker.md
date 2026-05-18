# Optional PDF processing worker

LexMind ships with a polling-based `DatabaseWatcher` that runs inside the
FastAPI process. For single-host development that's fine, but in
production it has the usual problems of in-process workers:

- the API and the heavy GenAI vectorisation share CPU / memory,
- crashing the watcher leaves uploaded PDFs stuck in `status="uploaded"`,
- there's no retry, no dead-letter, no horizontal scaling.

This document describes the optional, opt-in [arq](https://arq-docs.helpmanual.io/)
worker that solves those problems while staying compatible with the
existing watcher.

## Architecture

```
            ┌─────────────────────────┐
            │  POST /chat/upload      │
            └────────────┬────────────┘
                         │
              save_pdf_to_mongo(temp)
                         │
        ┌────────────────┼──────────────────┐
        │                │                  │
  status="uploaded"      │                  │
        │                │                  │
        │     ENABLE_PDF_WORKER=true        │
        │     enqueue_pdf_processing(_id)   │
        │                │                  │
        ▼                ▼                  ▼
 ┌──────────────┐  ┌──────────────┐   ┌────────────────┐
 │ DatabaseWatcher│ │ arq worker   │   │ Manual reprocess
 │  (fallback)  │  │ process_pdf  │   │ (e.g. migration)
 └──────┬───────┘  └──────┬───────┘   └────────┬───────┘
        │                 │                    │
        └─────────────────┼────────────────────┘
                          │
              ALL paths funnel through
        DatabaseWatcher._process_single_file
            (single source of truth)
                          │
                  status="processed"
```

Both paths use `find_one_and_update({status:"uploaded"}, {$set:{status:"processing"}})`
to atomically claim a row, so it is **safe** to enable the worker while
keeping the watcher running — they will not double-process.

## Deploying the worker

The worker is just another Python process. On the worker host:

```bash
pip install -r requirements.txt -r requirements-worker.txt
arq chatbot.workers.pdf_worker.WorkerSettings
```

On the API host, set:

```bash
ENABLE_PDF_WORKER=true
```

That's it. The API will start pushing job ids to the `lexmind:pdf` queue
in Redis after every successful upload. The worker drains that queue.

The watcher still runs in the API process as a self-healing safety net:
if Redis is briefly unreachable when an upload happens, the enqueue is
skipped (logged at WARN level) and the watcher picks the row up on its
next poll. You can disable the watcher entirely by guarding
`app_watcher.start()` in `backend/main.py` on `not is_worker_enabled()`,
but the default is to leave both running because the cost of the watcher
when there are no `uploaded` rows is negligible (one `find` per 5 s).

## Tuning

`chatbot/workers/pdf_worker.py:WorkerSettings`:

- `max_jobs = 4` — concurrent jobs per worker. Increase if your GenAI
  quota and CPU allow.
- `job_timeout = 600` — seconds; large or scanned PDFs can take a
  while.
- `keep_result = 300` — seconds the job result is kept in Redis.
- `queue_name = "lexmind:pdf"` — match this if you want to inspect via
  `arq` CLI tools.

## Disabling

To turn the worker back off:

1. `ENABLE_PDF_WORKER=false` (or unset) on the API host.
2. Stop the `arq` worker process.

The watcher will continue processing uploads. Any jobs still queued in
Redis will be drained the next time you bring the worker back; if you
prefer to discard them, `FLUSHDB` the worker DB or use
`arq.connections.create_pool(...).flushdb()` carefully.
