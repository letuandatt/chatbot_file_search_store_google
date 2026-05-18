"""
Qdrant client wrapper for the two collections this app uses:

  * `LAW_COLLECTION` (offline-ingested law corpus, shared across users)
  * `USER_COLLECTION` (PDFs each user uploads at runtime)

Both collections store the same payload shape so a single query path
serves both — only the collection name and (for user uploads) a
`session_id` filter differ.

Payload schema (per point):

    {
        "text":         "...",                # chunk content
        "source_file":  "Luat-Doanh-Nghiep.pdf",
        "page":         12,                    # 1-indexed PDF page
        "bbox":         [x0, y0, x1, y1],     # opendataloader-pdf bbox
        "section":      "Điều 7, Khoản 2",   # legal section label
        "doc_id":       "<mongo _id or law file uuid>",
        "session_id":   "<uuid>" | None,       # only set for user uploads
    }

The wrapper deliberately exposes a small, opinionated API rather than
leaking the Qdrant SDK to callers — that keeps it cheap to swap to
pgvector / Chroma / Weaviate later (only this file changes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from chatbot.config import config as app_config
from chatbot.core.embedder import EMBED_DIM

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One hit from a similarity search."""

    text: str
    score: float
    payload: dict = field(default_factory=dict)

    @property
    def section(self) -> str:
        return self.payload.get("section", "")

    @property
    def source_file(self) -> str:
        return self.payload.get("source_file", "")

    @property
    def page(self) -> int:
        return self.payload.get("page", 0)

    @property
    def citation(self) -> str:
        """Display string for use in answer footnotes."""
        bits: list[str] = []
        if self.section:
            bits.append(self.section)
        if self.source_file:
            bits.append(self.source_file)
        if self.page:
            bits.append(f"tr.{self.page}")
        return ", ".join(bits)


class QdrantVectorStore:
    """Owns the Qdrant client + collection lifecycle.

    Treat as a process-wide singleton; `AppContainer` instantiates one.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        law_collection: str | None = None,
        user_collection: str | None = None,
    ) -> None:
        self._url = url or app_config.QDRANT_URL
        self._api_key = api_key or app_config.QDRANT_API_KEY
        self._law_collection = law_collection or app_config.QDRANT_LAW_COLLECTION
        self._user_collection = user_collection or app_config.QDRANT_USER_COLLECTION
        self._client: QdrantClient | None = None

    # --- Connection -------------------------------------------------

    def _client_or_init(self) -> QdrantClient:
        if self._client is not None:
            return self._client
        if not self._url:
            raise RuntimeError("QDRANT_URL is not configured")
        # `prefer_grpc=False` keeps the deployment story simple — one
        # HTTP port to expose. Switch to grpc later if throughput
        # becomes a bottleneck.
        self._client = QdrantClient(url=self._url, api_key=self._api_key, prefer_grpc=False)
        return self._client

    @property
    def law_collection(self) -> str:
        return self._law_collection

    @property
    def user_collection(self) -> str:
        return self._user_collection

    # --- Schema -----------------------------------------------------

    def ensure_collections(self) -> None:
        """Create both collections if they don't exist. Idempotent."""
        client = self._client_or_init()
        for name in (self._law_collection, self._user_collection):
            try:
                client.get_collection(name)
                logger.info("[vectorstore] collection %s already exists", name)
            except Exception:
                logger.info("[vectorstore] creating collection %s", name)
                client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=EMBED_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                )

        # Filter perf: index session_id on the user collection so
        # `must={"session_id": ...}` lookups don't full-scan.
        try:
            client.create_payload_index(
                collection_name=self._user_collection,
                field_name="session_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Already indexed; that's fine.
            pass

    # --- Write path -------------------------------------------------

    def upsert(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: List[dict],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(payloads)):
            raise ValueError("ids/embeddings/payloads must be the same length")

        client = self._client_or_init()
        # Qdrant expects integer- or UUID-shaped point ids — pass our
        # deterministic string ids through unchanged. Qdrant accepts
        # strings here as of v1.5+.
        points = [
            qmodels.PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, embeddings, payloads)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)

    def delete_by_filter(self, collection: str, filter_: dict[str, Any]) -> int:
        """Remove every point whose payload matches *filter_* exactly."""
        client = self._client_or_init()
        qf = _build_filter(filter_)
        res = client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=qf),
            wait=True,
        )
        # Qdrant doesn't return the count of deleted points; this is best-effort.
        return getattr(res, "operation_id", 0) or 0

    # --- Read path --------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 20,
        filter_: Optional[dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        client = self._client_or_init()
        qf = _build_filter(filter_) if filter_ else None
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qf,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                text=(h.payload or {}).get("text", ""),
                score=h.score,
                payload=dict(h.payload or {}),
            )
            for h in hits
        ]


def _build_filter(filter_: dict[str, Any]) -> qmodels.Filter:
    """Translate a flat `{field: value}` dict into a Qdrant must-filter."""
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=k, match=qmodels.MatchValue(value=v)
            )
            for k, v in filter_.items()
        ]
    )


def chunks_to_points(
    chunks: Iterable[Any],
    embeddings: List[List[float]],
    *,
    extra_payload: Optional[dict[str, Any]] = None,
) -> tuple[List[str], List[List[float]], List[dict]]:
    """Helper used by both ingestion scripts to produce parallel
    `(ids, embeddings, payloads)` lists for `upsert()`.

    Expects each chunk to expose `.chunk_id`, `.text`, `.source_file`,
    `.page`, `.bbox`, `.section`, `.doc_id` (LegalChunk shape from
    `chatbot.core.chunker`).
    """
    chunk_list = list(chunks)
    if len(chunk_list) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    ids: list[str] = []
    payloads: list[dict] = []
    for c in chunk_list:
        ids.append(c.chunk_id)
        payload: dict[str, Any] = {
            "text": c.text,
            "source_file": c.source_file,
            "page": c.page,
            "bbox": list(c.bbox),
            "section": c.section,
            "doc_id": c.doc_id,
        }
        if extra_payload:
            payload.update(extra_payload)
        payloads.append(payload)

    return ids, embeddings, payloads
