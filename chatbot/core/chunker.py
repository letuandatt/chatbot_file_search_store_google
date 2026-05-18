"""
Legal-aware chunker for Vietnamese law PDFs.

Drives every step of the new ingestion pipeline:

    PDF -> opendataloader_pdf.convert(format='json') -> chunk_legal_document(...) -> embed -> Qdrant

The output of opendataloader-pdf is a hierarchical tree of `kids`
nodes. Each node carries `type`, `page number`, `bounding box`, and
(for text blocks) a `content` string. We walk this tree depth-first
to produce a flat reading-order sequence of blocks, then group blocks
into semantically meaningful **legal chunks**:

    * One chunk per "Điều X" — the smallest unit a court / lawyer
      would cite.
    * If a single Điều is larger than `MAX_CHUNK_CHARS`, fall back to
      sliding-window split on paragraph boundaries (so a 30-page
      Điều doesn't become one giant chunk that exceeds the embedder
      input limit).
    * Blocks before the first detected `Điều` are joined into a
      "preamble" chunk so chapter headings and definitions don't
      get lost.

Each `LegalChunk` carries `section`, `page`, and `bbox` so retrieval
hits can cite back to the original PDF exactly — which is the whole
reason for self-hosting this pipeline.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


# `Điều 7`, `Điều 7.`, `Điều 7:`, `Điều 7a` etc. The lookbehind
# avoids matching `(Điều 7)` mid-paragraph, only line-leading
# occurrences after we've stripped whitespace.
ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-zA-Z]?)[\.:\s]", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC\d]+)", re.IGNORECASE)

# `1.`, `1)` at start of a paragraph
KHOAN_RE = re.compile(r"^(\d+)[\.\)]\s")
# `a)`, `a.` at start
DIEM_RE = re.compile(r"^([a-zA-Z])[\.\)]\s")

# Conservative bound: Cohere embed-multilingual-v3 accepts ~512 tokens
# per input. 1 token ≈ 2-3 Vietnamese chars; cap at 1500 chars to
# stay safely below the limit while still containing whole sections.
MAX_CHUNK_CHARS = 1500
# Minimum useful chunk size — below this we'd just waste an embedding
# slot on something like a 2-word header.
MIN_CHUNK_CHARS = 40


@dataclass
class LegalChunk:
    """One indexable unit of a parsed legal document."""

    text: str
    source_file: str
    page: int
    bbox: tuple[float, float, float, float]
    section: str = ""
    chapter: str = ""
    doc_id: str = ""
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            # Deterministic id so re-ingesting the same PDF replaces
            # rather than duplicates rows.
            digest = hashlib.sha1(
                f"{self.doc_id or self.source_file}|{self.section}|{self.page}|{self.text[:100]}".encode(
                    "utf-8"
                )
            ).hexdigest()
            self.chunk_id = digest[:32]


@dataclass
class _Block:
    """A text-bearing block from opendataloader-pdf, flattened."""

    text: str
    page: int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------
# JSON walker
# --------------------------------------------------------------------

_TEXT_TYPES = {"heading", "paragraph", "list item", "table cell"}


def _flatten(node: Any) -> Iterable[_Block]:
    """Depth-first walk of an opendataloader-pdf JSON node.

    Yields one `_Block` per text-bearing leaf, preserving document
    reading order. Tables are flattened cell-by-cell — good enough for
    the law-corpus tables we care about (definition lists, schedule
    tables); high-fidelity table understanding is a future PR.
    """
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    content = node.get("content")

    if node_type in _TEXT_TYPES and isinstance(content, str):
        text = content.strip()
        if text:
            page = node.get("page number", 0) or 0
            bb = node.get("bounding box") or [0.0, 0.0, 0.0, 0.0]
            yield _Block(
                text=text,
                page=int(page),
                bbox=tuple(float(x) for x in bb[:4]),
            )

    # Descend into nested lists and tables
    for child_key in ("kids", "list items", "rows"):
        for child in node.get(child_key, []) or []:
            yield from _flatten(child)
    for row in node.get("cells", []) or []:
        yield from _flatten(row)


def _walk_root(doc_json: dict) -> List[_Block]:
    """Public-ish helper exposing the flat block list for testing."""
    blocks: list[_Block] = []
    for child in doc_json.get("kids", []) or []:
        blocks.extend(_flatten(child))
    return blocks


# --------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------


def _bbox_union(boxes: List[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _detect_section(text: str) -> Optional[str]:
    m = ARTICLE_RE.match(text)
    if m:
        return f"Điều {m.group(1)}"
    return None


def _detect_chapter(text: str) -> Optional[str]:
    m = CHAPTER_RE.match(text)
    if m:
        return f"Chương {m.group(1).upper()}"
    return None


def _split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split a too-long section on `Khoản` boundaries; fall back to paragraph splits."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    # Try `\n` paragraph splitting first
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 1 > max_chars and current:
            pieces.append("\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_legal_document(
    doc_json: dict,
    *,
    source_file: str,
    doc_id: str = "",
) -> List[LegalChunk]:
    """Group a parsed-PDF JSON tree into `LegalChunk` instances.

    `doc_json` is the dict produced by `opendataloader_pdf.convert(...,
    format='json')`. `source_file` is used both for citation display
    and as a fallback chunk-id seed when `doc_id` is empty.
    """
    blocks = _walk_root(doc_json)
    if not blocks:
        return []

    chunks: list[LegalChunk] = []
    current_section = ""
    current_chapter = ""
    current_blocks: list[_Block] = []

    def flush() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        joined = "\n".join(b.text for b in current_blocks).strip()
        if len(joined) < MIN_CHUNK_CHARS:
            current_blocks = []
            return
        for piece in _split_oversized(joined):
            chunks.append(
                LegalChunk(
                    text=piece,
                    source_file=source_file,
                    page=current_blocks[0].page,
                    bbox=_bbox_union([b.bbox for b in current_blocks]),
                    section=current_section,
                    chapter=current_chapter,
                    doc_id=doc_id,
                )
            )
        current_blocks = []

    for block in blocks:
        chap = _detect_chapter(block.text)
        if chap:
            flush()
            current_chapter = chap
            # Don't include chapter heading as its own chunk; absorb
            # into next section.
            continue

        section = _detect_section(block.text)
        if section:
            flush()
            current_section = section
            current_blocks = [block]
            continue

        current_blocks.append(block)

    flush()
    return chunks
