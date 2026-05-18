"""
Tool: tìm kiếm trong các file PDF user đã upload trong phiên hiện tại.

Each user upload lives in the shared `user_uploaded` Qdrant collection;
session isolation is enforced by a `session_id` payload filter so two
users (or two sessions of the same user) never see each other's chunks.
"""
from langchain_core.tools import StructuredTool

from chatbot.core.cache import app_cache
from chatbot.core.file_store import get_session_doc_ids
from chatbot.core.utils import safe_json_parse


def build_tool_search_uploaded(rag_pipeline):
    """Factory; binds the pipeline (with embedder + vector_store) into scope."""

    def search_uploaded_logic(query: str = None, session_id: str = None, **kwargs):
        # The supervisor sometimes passes args as a JSON blob in `query`.
        q_in = query if query else kwargs.get("query")
        parsed = safe_json_parse(q_in)
        if isinstance(parsed, dict) and parsed.get("query"):
            q_in = parsed.get("query")
            if parsed.get("session_id"):
                session_id = parsed.get("session_id")

        if not q_in or not session_id:
            return "Thiếu query hoặc session_id."

        # Sanity: only search if this session has at least one processed doc.
        # We don't need the IDs to filter (we use session_id payload), but the
        # check guards against the "no file uploaded yet" UX edge case.
        doc_ids = get_session_doc_ids(session_id)
        if not doc_ids:
            return "Chưa có file nào trong phiên này."

        cache_k = app_cache.generate_key("file", session_id, q_in)
        cached = app_cache.get(cache_k)
        if cached:
            return cached

        try:
            collection = rag_pipeline.vector_store.user_collection
            result = rag_pipeline.run_pipeline(
                original_query=str(q_in),
                collection=collection,
                filter_={"session_id": session_id},
            )
            app_cache.set(cache_k, result, ttl=1800)
            return result
        except Exception as exc:
            return f"Lỗi tra cứu file: {exc}"

    return StructuredTool.from_function(
        func=search_uploaded_logic,
        name="tool_search_uploaded_file",
        description="Tìm kiếm trong tài liệu upload (PDF).",
    )
