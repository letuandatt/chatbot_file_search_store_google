from langchain_core.tools import StructuredTool
from chatbot.core.db import DB_DOCUMENTS_COLLECTION
from chatbot.core.file_store import get_session_file_stores
from chatbot.core.cache import app_cache
from chatbot.core.utils import safe_json_parse


def _describe_pending_files(session_id: str) -> str:
    """
    When `get_session_file_stores` returns nothing, look up the raw
    documents collection to tell the user *why* — still processing
    vs. genuinely no upload — instead of returning the same blanket
    message in both cases.
    """
    if DB_DOCUMENTS_COLLECTION is None:
        return "Chưa có file nào trong phiên này."
    try:
        statuses = [
            doc.get("status")
            for doc in DB_DOCUMENTS_COLLECTION.find(
                {"session_id": session_id}, {"status": 1}
            )
        ]
    except Exception:
        return "Chưa có file nào trong phiên này."

    if not statuses:
        return "Chưa có file nào trong phiên này."
    if any(s in ("uploaded", "processing") for s in statuses):
        return (
            "File của bạn đang được hệ thống xử lý. "
            "Vui lòng thử lại sau khoảng 30 giây."
        )
    if all(s in ("error", "error_processing") for s in statuses):
        return (
            "Hệ thống không xử lý được file bạn đã tải lên. "
            "Vui lòng thử tải lại."
        )
    return "Chưa có file nào trong phiên này."


def build_tool_search_uploaded(rag_pipeline, genai_client):
    """
    Factory function:
    - rag_pipeline: Dùng để search & rerank.
    - genai_client: Dùng để kiểm tra xem Store có còn sống không (Verify).
    """

    def _validate_stores(stores: list[str]) -> list[str]:
        """
        Helper: Kiểm tra danh sách store_name có thực sự tồn tại trên Google không.
        """
        valid_stores = []
        if not genai_client:
            return stores  # Fallback nếu client chưa sẵn sàng

        for name in stores:
            try:
                # Thử lấy thông tin store. Nếu store bị xóa/lỗi -> Ném Exception
                genai_client.file_search_stores.get(name=name)
                valid_stores.append(name)
            except Exception as e:
                print(f"[ToolUpload] Store '{name}' không truy cập được (đã bị xóa?): {e}")
                # Không thêm vào valid_stores
        return valid_stores

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
            return _describe_pending_files(session_id)

        valid_stores = _validate_stores(user_stores)

        if not valid_stores:
            return "Các file trong phiên này không còn khả dụng (có thể đã bị xóa hoặc hết hạn)."

        cache_k = app_cache.generate_key("file", session_id, q_in)
        cached = app_cache.get(cache_k)
        if cached:
            return cached

        try:
            result = rag_pipeline.run_pipeline(
                original_query=str(q_in),
                store_names=valid_stores
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