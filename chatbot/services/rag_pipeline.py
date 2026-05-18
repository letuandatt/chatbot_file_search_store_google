"""
Advanced RAG pipeline (CRAG-style) over the self-hosted Qdrant store.

The previous implementation outsourced retrieval to Google's managed
`file_search_stores` — which meant the rerank + evaluator + generate
stages downstream were stuck with Google's chunking and citation
granularity. We now own retrieval end-to-end:

    query
      └─► QueryGenerator (multi-query)                      [LLM]
            └─► QdrantVectorStore.search(top_k=20)          [vector]
                  └─► CohereReranker.rerank(top_n=5)        [Cohere]
                        └─► RelevanceEvaluator (per chunk)  [LLM]
                              └─► Gemini generate (final)   [LLM]

Each `RetrievedChunk` carries `section` (`Điều X`), `source_file`,
`page` and `bbox` so the final answer can cite back to the exact PDF
page — the whole reason for self-hosting this.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from chatbot.config import config as app_config
from chatbot.core.evaluator import RelevanceEvaluator
from chatbot.core.query_generator import QueryGenerator
from chatbot.core.reranker import CohereReranker
from chatbot.core.vectorstore import QdrantVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


class AdvancedRagPipeline:
    """CRAG pipeline that retrieves from a Qdrant collection."""

    def __init__(
        self,
        genai_client: Any,
        text_llm_langchain: Any,
        embedder: Any,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.client = genai_client
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = CohereReranker()
        self.query_gen = QueryGenerator(text_llm_langchain)
        self.evaluator = RelevanceEvaluator(text_llm_langchain)
        self.model_name = app_config.TEXT_MODEL_NAME

    # --- Retrieval helper -------------------------------------------

    def _fetch_chunks(
        self,
        query: str,
        collection: str,
        filter_: Optional[dict[str, Any]] = None,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Embed the query and run a similarity search."""
        try:
            qvec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("[Pipeline] embed_query failed: %s", exc)
            return []

        try:
            return self.vector_store.search(
                collection=collection,
                query_vector=qvec,
                top_k=top_k,
                filter_=filter_ or None,
            )
        except Exception as exc:
            logger.warning("[Pipeline] vector_store.search failed: %s", exc)
            return []

    # --- Public API -------------------------------------------------

    def run_pipeline(
        self,
        original_query: str,
        collection: str,
        filter_: Optional[dict[str, Any]] = None,
    ) -> str:
        """Multi-query CRAG retrieve → rerank → evaluate → generate.

        `collection` selects the Qdrant collection (law vs user).
        `filter_` further narrows the result set (e.g. `{"session_id":
        x}` for user uploads).
        """
        queries = self.query_gen.generate_queries(original_query)
        logger.info("[Pipeline] generated queries=%s", queries)

        final_chunks: list[RetrievedChunk] = []

        for q in queries:
            logger.debug("[Pipeline] trying query: %s", q)

            raw_chunks = self._fetch_chunks(q, collection, filter_)
            if not raw_chunks:
                continue

            # Rerank — give the reranker the raw text only; we re-attach
            # the metadata after by mapping reranked text back to chunks.
            texts = [c.text for c in raw_chunks]
            by_text = {c.text: c for c in raw_chunks}
            top_texts = self.reranker.rerank(q, list(set(texts)), top_n=5)
            top_chunks = [by_text[t] for t in top_texts if t in by_text]

            good_chunks_in_pass: list[RetrievedChunk] = []
            for chunk in top_chunks:
                grade = self.evaluator.evaluate(original_query, chunk.text)
                if grade == "YES":
                    good_chunks_in_pass.append(chunk)
                else:
                    logger.debug("[Evaluator] rejected chunk for query '%s'", q)

            if good_chunks_in_pass:
                final_chunks.extend(good_chunks_in_pass)
                if len(final_chunks) >= 2:
                    break
            else:
                logger.debug("[Pipeline] query '%s' yielded no relevant info", q)

        if not final_chunks:
            return (
                "Xin lỗi, tôi đã thử tìm kiếm trong tài liệu nhưng không thấy "
                "thông tin liên quan đến câu hỏi của bạn. "
                "(CRAG: No relevant docs found)"
            )

        # Deduplicate while preserving citation metadata
        seen: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for c in final_chunks:
            if c.text not in seen:
                seen.add(c.text)
                unique_chunks.append(c)

        # Build context with citation markers so the LLM can cite back
        context_blocks = [
            f"[{c.citation or 'không rõ nguồn'}]\n{c.text}" for c in unique_chunks
        ]
        context_text = "\n\n---\n\n".join(context_blocks)

        final_prompt = f"""Dựa vào các thông tin ĐÃ ĐƯỢC KIỂM CHỨNG sau đây, hãy trả lời câu hỏi.

QUY TẮC TRÍCH DẪN (BẮT BUỘC):
1. Trích dẫn RÕ RÀNG: Điều số mấy, Khoản số mấy, Điểm nào (nếu context có ghi)
2. Ghi rõ tên văn bản pháp luật và số hiệu (ví dụ: Chỉ thị 12/CT-TTg năm 2022)
3. Nếu context KHÔNG ghi rõ Điều/Khoản, chỉ trích dẫn tên văn bản và số hiệu
4. TUYỆT ĐỐI KHÔNG tự bịa ra số Điều/Khoản nếu context không ghi rõ
5. Kèm tên file nguồn (in trong dấu ngoặc vuông trước mỗi đoạn) ở cuối câu trả lời như: "(Nguồn: …)"

NGỮ CẢNH (CONTEXT):
{context_text}

CÂU HỎI:
{original_query}
"""
        try:
            res = self.client.models.generate_content(
                model=self.model_name, contents=final_prompt
            )
            return res.text
        except Exception as exc:
            return f"Lỗi tổng hợp: {exc}"
