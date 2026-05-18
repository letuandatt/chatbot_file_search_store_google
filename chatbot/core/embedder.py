"""
Embedding wrapper for Cohere `embed-multilingual-v3`.

The model returns 1024-dimensional float vectors. We use this **single
vendor** for both document ingestion (offline) and query embedding
(runtime) so the vector space is consistent — mixing embedders would
make cosine similarity meaningless.

Cohere has two input types we care about:

    * `search_document`: optimised for the documents being indexed
    * `search_query`:    optimised for the user query at retrieval time

Mixing them up degrades recall significantly, so the helpers below
expose them as two distinct methods rather than a single `embed(text)`.

The wrapper is intentionally tiny — it just batches requests and
swallows transient errors so a long ingestion run doesn't die on a
single rate-limit hiccup. Retries are linear (no exponential backoff)
because Cohere's published rate limit is well under what a sane
ingestion job will produce; if you start hitting it, lower the batch
size instead.
"""
from __future__ import annotations

import logging
import time
from typing import List, Sequence

import cohere

from chatbot.config import config as app_config

logger = logging.getLogger(__name__)

# Cohere's embed endpoint accepts up to 96 texts per call. We stay
# slightly under to give headroom for very long inputs that might be
# rejected for character count.
_BATCH_SIZE = 64

# The model output is 1024-dimensional. Stored as a constant so the
# vector store can declare collection geometry without round-tripping
# Cohere.
EMBED_DIM = 1024
EMBED_MODEL = "embed-multilingual-v3.0"


class CohereEmbedder:
    """Thin wrapper around `cohere.ClientV2.embed()`.

    The client is created lazily on first use so importing this module
    in environments without `COHERE_API_KEY` (tests, migrations) does
    not explode.
    """

    def __init__(self, api_key: str | None = None, model: str = EMBED_MODEL):
        self._api_key = api_key or app_config.COHERE_API_KEY
        self._model = model
        self._client: cohere.ClientV2 | None = None

    def _client_or_init(self) -> cohere.ClientV2:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not configured; cannot use the Cohere embedder."
            )
        self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def _embed(self, texts: Sequence[str], input_type: str) -> List[List[float]]:
        """Shared batching path used by both query/document helpers."""
        if not texts:
            return []

        client = self._client_or_init()
        all_vectors: List[List[float]] = []

        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            for attempt in range(3):
                try:
                    resp = client.embed(
                        texts=batch,
                        model=self._model,
                        input_type=input_type,
                        embedding_types=["float"],
                    )
                    all_vectors.extend(resp.embeddings.float_)
                    break
                except Exception as exc:  # pragma: no cover - depends on remote
                    logger.warning(
                        "[embedder] batch %d attempt %d failed: %s",
                        start // _BATCH_SIZE,
                        attempt + 1,
                        exc,
                    )
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))

        return all_vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed chunks that will live in the vector store (offline + worker)."""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query at retrieval time."""
        vecs = self._embed([text], input_type="search_query")
        if not vecs:
            raise RuntimeError("Cohere returned no embedding for the query")
        return vecs[0]
