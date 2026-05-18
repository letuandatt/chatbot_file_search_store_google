# Self-hosted RAG pipeline

This document describes how the law-document retrieval pipeline works
after PR R. Prior to PR R both the corpus ingestion and the runtime
retrieval were delegated to Google's managed `file_search_stores` —
the new pipeline is fully self-hosted with `opendataloader-pdf`,
Cohere embeddings, and a local Qdrant instance.

## Why move off `file_search_stores`?

| Concern | Google `file_search_stores` | Self-hosted |
|--------|-----------------------------|-------------|
| Citation granularity | filename only | `Điều X, Khoản Y, page Z, bbox` |
| Chunking control | opaque, changes with model updates | deterministic, legal-aware |
| OCR for scanned PDFs | varies | always on (Java backend) |
| Cost | per-query + storage | one-time embed (~$5 for current corpus) |
| Lock-in | Google ecosystem | swap any layer |

The valuable downstream stages (Cohere rerank, LLM relevance
evaluator, Gemini generation) are unchanged — only retrieval moved.

## Architecture

```
ingestion (one-off)               runtime (every query)
──────────────────────────        ──────────────────────────
DATA_DIR/*.pdf                    user query
  │                                 │
  ▼                                 ▼
opendataloader-pdf  ──►  JSON    embed_query()  ──►  vector
  │                                 │                   │
  ▼                                 ▼                   ▼
chunk_legal_document             QdrantVectorStore.search(top_k=20)
  │                                 │
  ▼                                 ▼
embed_documents() ───►  vectors  CohereReranker.rerank(top_n=5)
  │                                 │
  ▼                                 ▼
QdrantVectorStore.upsert()       RelevanceEvaluator (per-chunk YES/NO)
                                    │
                                    ▼
                                 Gemini generate (with citations)
```

## Components

| Module | Purpose |
|--------|---------|
| `chatbot/core/embedder.py` | Wraps Cohere `embed-multilingual-v3` (1024-d). Two input modes: `embed_documents` (offline) + `embed_query` (runtime). |
| `chatbot/core/chunker.py` | Walks the opendataloader-pdf JSON tree and groups text blocks into legal-aware chunks (one chunk per `Điều X`; oversized articles auto-split). |
| `chatbot/core/vectorstore.py` | Thin wrapper around the Qdrant Python client. Owns two collections: `law_corpus` (shared) and `user_uploaded` (filtered by `session_id`). |
| `chatbot/services/rag_pipeline.py` | CRAG-style pipeline that retrieves from a Qdrant collection, reranks, evaluates, and generates. |
| `chatbot/setup_main_store/setup_main_store.py` | Offline CLI that ingests `$DATA_DIR/*.pdf` into the `law_corpus` collection. |
| `chatbot/setup_main_store/test_query.py` | Offline CLI that exercises retrieval (and optionally the LLM stage) for smoke testing. |
| `chatbot/core/file_store.py` | Runtime `/chat/upload` path. After save-to-GridFS, the background worker calls `process_and_vectorize_pdf` to chunk + embed + upsert into `user_uploaded`. |
| `scripts/migrate_uploaded_to_qdrant.py` | One-shot migration script for existing user uploads that were processed under the old Google pipeline. |

## Prerequisites

1. **Java 11+** — required by opendataloader-pdf.
   ```
   sudo apt-get install -y openjdk-17-jre-headless
   java -version
   ```
2. **Qdrant** running locally:
   ```
   docker compose up -d qdrant
   ```
   The compose file binds it to `127.0.0.1:6333`. For production, set
   `QDRANT_URL` to your cluster URL and `QDRANT_API_KEY` to the secret.
3. **Cohere API key** — same key already used by the reranker. Make
   sure `COHERE_API_KEY` is set in `chatbot/.env`.
4. Python deps:
   ```
   pip install -r requirements.txt
   ```

## First-time corpus ingestion

```
# 1. Drop PDFs under chatbot/data/CongThongTinDienTu/ (or any folder)

# 2. Verify chunking without burning embed credits
python -m chatbot.setup_main_store.setup_main_store --dry-run

# 3. Run the real ingestion (one-time)
python -m chatbot.setup_main_store.setup_main_store

# 4. Smoke-test retrieval
python -m chatbot.setup_main_store.test_query --retrieve-only \
    "Điều kiện thành lập công ty TNHH một thành viên?"

# 5. End-to-end (retrieve + rerank + evaluate + generate)
python -m chatbot.setup_main_store.test_query \
    "Cổ đông sáng lập có những quyền gì?"
```

Re-running the ingestion is **safe and idempotent**: chunk ids are
deterministic from `(source_file, section, page, text-prefix)`, so a
re-run upserts in place rather than duplicating rows.

## Migrating old user uploads

For deployments that ran the old Google-based pipeline:

```
# Optionally narrow to one user during testing
python -m scripts.migrate_uploaded_to_qdrant --user-id <uid>

# Full migration
python -m scripts.migrate_uploaded_to_qdrant
```

The script rebuilds chunks from the GridFS-stored binary, embeds them,
and upserts into `user_uploaded`. Old `file_store_name` Mongo fields
are left in place so a rollback is possible.

## Tuning knobs

| Where | Default | Notes |
|-------|---------|-------|
| `chunker.MAX_CHUNK_CHARS` | 1500 | Stays under Cohere's 512-token limit. Raise cautiously. |
| `setup_main_store --batch-size` | 32 | Embed batch size. Lower if you hit Cohere rate limits. |
| `top_k` (search) | 20 | Pulled to the reranker; reranker keeps top 5. |
| `app_cache TTL` for law | 3600s | Long because law text doesn't change often. |
| `app_cache TTL` for user uploads | 1800s | Shorter — user may delete the file. |

## Operational notes

- **Atomicity**: the upload endpoint returns as soon as the binary is
  in GridFS. The watcher (or the optional arq worker — see PR #7)
  picks up `status=uploaded` rows and runs the embed pipeline
  asynchronously. The front-end polls `/chat/file/{file_id}/status`
  for `processed` before letting the user query against the file.
- **Failure modes**: if `opendataloader-pdf` crashes on a malformed
  PDF, the document is marked `status=error_processing` with the
  exception message recorded in `error`. The user sees a clear error
  in the UI; the rest of the queue keeps draining.
- **Session isolation**: user uploads share a single collection, but
  every search call adds `must={"session_id": <id>}` so two users
  never see each other's chunks. There is also a Qdrant payload index
  on `session_id` so the filter is O(log n).
- **Citations**: every chunk payload carries `section`, `source_file`,
  `page`, and `bbox`. The LLM prompt instructs the model to cite each
  retrieved block; the JSON `bbox` is available client-side if/when
  you add a "highlight in PDF" feature.
