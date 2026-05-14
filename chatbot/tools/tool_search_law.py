"""
Tool: tra cứu văn bản pháp luật.

Replaces the old Google `file_search_stores` integration with a query
against the self-hosted Qdrant law collection. Rerank, evaluator, and
generate stages remain in `AdvancedRagPipeline` — only retrieval moved.
"""
from langchain_core.tools import StructuredTool

from chatbot.core.cache import app_cache


def build_tool_search_law(rag_pipeline):
    """Factory; binds the pipeline (with embedder + vector_store) into scope."""

    def search_law_logic(query: str) -> str:
        """Tra cứu thông tin trong văn bản quy phạm pháp luật."""
        if not query or not query.strip():
            return "Thiếu nội dung câu hỏi."

        cache_k = app_cache.generate_key("law", "adv", query)
        cached = app_cache.get(cache_k)
        if cached:
            return cached

        try:
            collection = rag_pipeline.vector_store.law_collection
            result = rag_pipeline.run_pipeline(
                original_query=query,
                collection=collection,
            )
            app_cache.set(cache_k, result, ttl=3600)
            return result
        except Exception as exc:
            return f"Lỗi khi tra cứu: {exc}"

    return StructuredTool.from_function(
        func=search_law_logic,
        name="tool_search_law",
        description="Tra cứu văn bản quy phạm pháp luật",
    )
