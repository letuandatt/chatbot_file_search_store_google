"""Unit tests for the Qdrant wrapper that don't require a live Qdrant.

We mock the SDK client; the goal is to exercise our argument-marshaling
code (filter construction, payload shape, chunk-to-point conversion).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("JWT_SECRET_KEY", "test-only")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from chatbot.core.chunker import LegalChunk  # noqa: E402
from chatbot.core.vectorstore import (  # noqa: E402
    QdrantVectorStore,
    RetrievedChunk,
    _build_filter,
    chunks_to_points,
)


def _chunk(text: str, **overrides) -> LegalChunk:
    base = {
        "text": text,
        "source_file": "law.pdf",
        "page": 1,
        "bbox": (0.0, 0.0, 0.0, 0.0),
        "section": "Điều 1",
        "doc_id": "L1",
    }
    base.update(overrides)
    return LegalChunk(**base)


class TestRetrievedChunk:
    def test_citation_all_fields(self):
        c = RetrievedChunk(
            text="X",
            score=0.9,
            payload={
                "section": "Điều 7",
                "source_file": "DN2020.pdf",
                "page": 12,
            },
        )
        assert c.citation == "Điều 7, DN2020.pdf, tr.12"

    def test_citation_missing_fields(self):
        c = RetrievedChunk(text="X", score=0.9, payload={"source_file": "x.pdf"})
        assert c.citation == "x.pdf"

    def test_citation_empty(self):
        assert RetrievedChunk(text="X", score=0.9, payload={}).citation == ""


class TestChunksToPoints:
    def test_basic_shape(self):
        chunks = [_chunk("Điều 1 nội dung A"), _chunk("Điều 2 nội dung B", section="Điều 2")]
        embeddings = [[0.1] * 4, [0.2] * 4]

        ids, vecs, payloads = chunks_to_points(chunks, embeddings)

        assert len(ids) == len(vecs) == len(payloads) == 2
        assert ids[0] != ids[1]
        # Payload carries everything the search path needs for citation
        assert payloads[0]["text"] == "Điều 1 nội dung A"
        assert payloads[0]["section"] == "Điều 1"
        assert payloads[0]["source_file"] == "law.pdf"
        assert payloads[0]["page"] == 1
        assert payloads[0]["doc_id"] == "L1"
        assert payloads[0]["bbox"] == [0.0, 0.0, 0.0, 0.0]

    def test_extra_payload_merged(self):
        chunks = [_chunk("foo")]
        ids, vecs, payloads = chunks_to_points(
            chunks, [[0.0] * 4], extra_payload={"session_id": "sess-1", "user_id": "u-1"}
        )
        assert payloads[0]["session_id"] == "sess-1"
        assert payloads[0]["user_id"] == "u-1"

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            chunks_to_points([_chunk("a"), _chunk("b")], [[0.0]])


class TestFilterBuilder:
    def test_single_field(self):
        f = _build_filter({"session_id": "abc"})
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "session_id"
        assert cond.match.value == "abc"

    def test_multiple_fields(self):
        f = _build_filter({"session_id": "abc", "user_id": "u1"})
        assert len(f.must) == 2


class TestQdrantVectorStore:
    def test_upsert_passes_through_to_sdk(self):
        store = QdrantVectorStore(url="http://qdrant:6333")
        mock_client = MagicMock()
        store._client = mock_client

        store.upsert(
            collection="law_corpus",
            ids=["a", "b"],
            embeddings=[[0.1] * 4, [0.2] * 4],
            payloads=[{"text": "x"}, {"text": "y"}],
        )

        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "law_corpus"
        assert len(kwargs["points"]) == 2

    def test_upsert_validates_lengths(self):
        store = QdrantVectorStore(url="http://qdrant:6333")
        store._client = MagicMock()
        with pytest.raises(ValueError):
            store.upsert(
                collection="law_corpus", ids=["a"], embeddings=[], payloads=[{}]
            )

    def test_upsert_empty_is_noop(self):
        store = QdrantVectorStore(url="http://qdrant:6333")
        store._client = MagicMock()
        store.upsert(collection="x", ids=[], embeddings=[], payloads=[])
        store._client.upsert.assert_not_called()

    def test_search_translates_filter(self):
        store = QdrantVectorStore(url="http://qdrant:6333")
        mock_client = MagicMock()
        mock_client.search.return_value = []
        store._client = mock_client

        store.search(
            collection="user_uploaded",
            query_vector=[0.1] * 4,
            top_k=5,
            filter_={"session_id": "sess-1"},
        )

        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["collection_name"] == "user_uploaded"
        assert kwargs["limit"] == 5
        assert kwargs["query_filter"] is not None
        # The filter we passed is a Qdrant Filter object
        assert kwargs["query_filter"].must[0].key == "session_id"

    def test_search_returns_retrieved_chunks(self):
        store = QdrantVectorStore(url="http://qdrant:6333")
        mock_client = MagicMock()
        # Simulate Qdrant ScoredPoint shape
        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {"text": "Điều 7 ...", "source_file": "L.pdf", "page": 3}
        mock_client.search.return_value = [mock_hit]
        store._client = mock_client

        hits = store.search(collection="law_corpus", query_vector=[0.0] * 4, top_k=1)
        assert len(hits) == 1
        assert hits[0].text.startswith("Điều 7")
        assert hits[0].score == 0.85
        assert hits[0].source_file == "L.pdf"

    def test_collection_properties_expose_names(self):
        store = QdrantVectorStore(
            url="http://qdrant:6333",
            law_collection="my_law",
            user_collection="my_users",
        )
        assert store.law_collection == "my_law"
        assert store.user_collection == "my_users"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
