"""Unit tests for the legal-aware chunker.

These tests run with a synthetic opendataloader-pdf JSON tree so they
have no external deps — no Java, no Qdrant, no Cohere, no real PDFs.
"""
from __future__ import annotations

import os

import pytest

# Importing the chunker pulls in `chatbot.config` which expects a
# `JWT_SECRET_KEY` env var at import time. Keep the test self-contained.
os.environ.setdefault("JWT_SECRET_KEY", "test-only")

from chatbot.core.chunker import (  # noqa: E402  (after env setup)
    ARTICLE_RE,
    CHAPTER_RE,
    LegalChunk,
    chunk_legal_document,
)


def _para(text: str, page: int = 1, bbox=(0.0, 0.0, 0.0, 0.0)) -> dict:
    return {
        "type": "paragraph",
        "page number": page,
        "bounding box": list(bbox),
        "content": text,
    }


def _heading(text: str, page: int = 1, bbox=(0.0, 0.0, 0.0, 0.0)) -> dict:
    return {
        "type": "heading",
        "page number": page,
        "bounding box": list(bbox),
        "content": text,
    }


class TestRegexes:
    """Confidence checks on the legal-section regexes."""

    def test_article_basic(self):
        assert ARTICLE_RE.match("Điều 7. Phạm vi")

    def test_article_suffix_letter(self):
        # `Điều 5a`, `Điều 12b` etc. show up in amendments.
        assert ARTICLE_RE.match("Điều 5a. Bổ sung")

    def test_article_no_match_midline(self):
        assert ARTICLE_RE.match("Theo Điều 7, công ty...") is None

    def test_chapter_roman(self):
        assert CHAPTER_RE.match("Chương III: Tổ chức quản lý")

    def test_chapter_arabic(self):
        assert CHAPTER_RE.match("Chương 2 Điều khoản")


class TestChunkLegalDocument:
    def test_empty_doc_returns_empty(self):
        assert chunk_legal_document({"kids": []}, source_file="x.pdf") == []

    def test_single_article(self):
        doc = {
            "kids": [
                _heading("LUẬT DOANH NGHIỆP 2020", page=1, bbox=(10, 800, 400, 815)),
                _para("Điều 1. Phạm vi điều chỉnh", page=1, bbox=(10, 770, 400, 785)),
                _para("Luật này quy định về việc thành lập...", page=1, bbox=(10, 750, 400, 765)),
            ]
        }
        chunks = chunk_legal_document(doc, source_file="law.pdf", doc_id="law")
        assert len(chunks) == 1
        c = chunks[0]
        assert c.section == "Điều 1"
        assert c.source_file == "law.pdf"
        assert c.page == 1
        # bbox is union of contained blocks
        assert c.bbox == (10.0, 750.0, 400.0, 785.0)
        assert "Phạm vi điều chỉnh" in c.text
        assert "thành lập" in c.text

    def test_multiple_articles_split_by_section(self):
        doc = {
            "kids": [
                _para("Điều 1. Phạm vi điều chỉnh của luật này", page=1),
                _para("Luật này quy định về việc thành lập doanh nghiệp.", page=1),
                _para("Điều 2. Đối tượng áp dụng của luật này", page=2),
                _para("Doanh nghiệp và các cơ quan, tổ chức, cá nhân liên quan.", page=2),
            ]
        }
        chunks = chunk_legal_document(doc, source_file="law.pdf")
        assert [c.section for c in chunks] == ["Điều 1", "Điều 2"]
        assert chunks[0].page == 1
        assert chunks[1].page == 2

    def test_chapter_inherited(self):
        doc = {
            "kids": [
                _para("Chương I: Quy định chung", page=1),
                _para("Điều 1. Phạm vi điều chỉnh luật này", page=1),
                _para("Quy định chi tiết về việc thành lập.", page=1),
                _para("Chương II: Tổ chức doanh nghiệp", page=5),
                _para("Điều 10. Quyền của cổ đông sáng lập", page=5),
                _para("Cổ đông sáng lập có quyền biểu quyết về các vấn đề.", page=5),
            ]
        }
        chunks = chunk_legal_document(doc, source_file="law.pdf")
        assert len(chunks) == 2
        assert chunks[0].chapter == "Chương I"
        assert chunks[1].chapter == "Chương II"

    def test_chunk_id_is_deterministic(self):
        doc = {
            "kids": [
                _para("Điều 1. Quy định về phạm vi điều chỉnh của văn bản", page=1),
                _para("Nội dung chi tiết được quy định rõ ràng.", page=1),
            ]
        }
        a = chunk_legal_document(doc, source_file="law.pdf", doc_id="X")
        b = chunk_legal_document(doc, source_file="law.pdf", doc_id="X")
        assert len(a) == 1 and len(b) == 1
        assert a[0].chunk_id == b[0].chunk_id

    def test_chunk_id_changes_with_doc_id(self):
        doc = {
            "kids": [
                _para("Điều 1. Quy định về phạm vi điều chỉnh của văn bản", page=1),
                _para("Nội dung chi tiết được quy định rõ ràng.", page=1),
            ]
        }
        a = chunk_legal_document(doc, source_file="law.pdf", doc_id="X")
        b = chunk_legal_document(doc, source_file="law.pdf", doc_id="Y")
        assert a[0].chunk_id != b[0].chunk_id

    def test_oversized_article_is_split(self):
        # 1 article whose body crosses MAX_CHUNK_CHARS
        long_body = "\n".join(["Một đoạn văn dài " * 10 for _ in range(40)])
        doc = {
            "kids": [
                _para("Điều 7. Quy định dài", page=3),
                _para(long_body, page=3),
            ]
        }
        chunks = chunk_legal_document(doc, source_file="law.pdf")
        assert len(chunks) > 1
        # Every piece keeps the same section label
        assert all(c.section == "Điều 7" for c in chunks)

    def test_lists_are_flattened(self):
        doc = {
            "kids": [
                _para("Điều 2. Đối tượng áp dụng của văn bản này", page=1),
                {
                    "type": "list",
                    "page number": 1,
                    "bounding box": [0, 0, 0, 0],
                    "list items": [
                        {
                            "type": "list item",
                            "page number": 1,
                            "bounding box": [0, 0, 0, 0],
                            "content": "1. Doanh nghiệp được thành lập và hoạt động theo",
                        },
                        {
                            "type": "list item",
                            "page number": 1,
                            "bounding box": [0, 0, 0, 0],
                            "content": "2. Cá nhân và các cơ quan, tổ chức liên quan khác",
                        },
                    ],
                },
            ]
        }
        chunks = chunk_legal_document(doc, source_file="law.pdf")
        assert len(chunks) == 1
        assert "Doanh nghiệp" in chunks[0].text
        assert "Cá nhân" in chunks[0].text

    def test_minimum_chunk_length(self):
        # Tiny standalone preamble is dropped (below MIN_CHUNK_CHARS).
        doc = {"kids": [_para("Hi", page=1)]}
        assert chunk_legal_document(doc, source_file="law.pdf") == []


class TestLegalChunk:
    def test_post_init_generates_id(self):
        c = LegalChunk(
            text="Điều 1. Foo",
            source_file="x.pdf",
            page=1,
            bbox=(0, 0, 0, 0),
            section="Điều 1",
            doc_id="x",
        )
        assert c.chunk_id
        assert len(c.chunk_id) == 32

    def test_explicit_id_preserved(self):
        c = LegalChunk(
            text="Điều 1. Foo",
            source_file="x.pdf",
            page=1,
            bbox=(0, 0, 0, 0),
            section="Điều 1",
            doc_id="x",
            chunk_id="custom-id-123",
        )
        assert c.chunk_id == "custom-id-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
