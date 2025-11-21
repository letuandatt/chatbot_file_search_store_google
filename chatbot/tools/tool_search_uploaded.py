from langchain_core.tools import StructuredTool
from chatbot.core.file_store import get_session_file_stores
from chatbot.core.cache import app_cache
from chatbot.core.utils import safe_json_parse


def build_tool_search_uploaded(rag_pipeline, genai_client):
    # Vẫn cần genai_client để check store exists (optional), hoặc đưa logic check vào pipeline

    def search_uploaded_logic(query: str = None, session_id: str = None, **kwargs):
        q_in = query if query else kwargs.get("query")
        parsed = safe_json_parse(q_in)
        if isinstance(parsed, dict) and parsed.get("query"):
            q_in = parsed.get("query")
            if parsed.get("session_id"):
                session_id = parsed.get("session_id")

        if not q_in or not session_id:
            return "Thiếu query hoặc session_id."

        user_stores = get_session_file_stores(session_id)
        if not user_stores:
            return "Chưa có file nào trong phiên này."

        # (Optional) Verify stores exist logic here...

        cache_k = app_cache.generate_key("file", session_id, q_in)
        cached = app_cache.get(cache_k)
        if cached:
            return cached

        try:
            result = rag_pipeline.run_pipeline(
                original_query=str(q_in),
                store_names=user_stores
            )
            app_cache.set(cache_k, result, ttl=1800)
            return result
        except Exception as e:
            return f"Lỗi tra cứu file: {e}"

    return StructuredTool.from_function(
        func=search_uploaded_logic,
        name="tool_search_uploaded_file",
        description="Tìm kiếm trong tài liệu upload (PDF)."
    )