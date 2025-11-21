from langchain_core.tools import tool
from chatbot.core.db import DB_DOCUMENTS_COLLECTION
from chatbot.core.utils import cache_key, get_cache, set_cache

@tool
def tool_list_uploaded_files(user_id: str = None):
    """
    List all files uploaded by the given user (across sessions).
    This avoids session-mismatch issues in CLI dev flow.
    Returns a newline-separated list or an inormative message.
    """
    if not user_id:
        return "Lỗi: thiếu user_id."

    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return "Lỗi DB: documents collection chưa sẵn sàng."

    cache_k = cache_key("listfiles", user_id, "list")
    cached = get_cache(cache_k)
    if cached:
        return cached

    try:
        cursor = coll.find({"user_id": user_id}, {"filename": 1, "status": 1, "session_id": 1}).sort("created_at", -1)
        files = []
        for d in cursor:
            fn = d.get("filename") or "<unknown>"
            st = d.get("status") or "unknown"
            sid = d.get("session_id") or "unknown"
            files.append(f"{fn} (status={st}, session={sid})")
        if not files:
            out = "Bạn chưa upload file nào."
            set_cache(cache_k, out)
            return out
        out = "\n".join(files)
        set_cache(cache_k, out)
        return out
    except Exception as e:
        return f"Lỗi khi truy vấn danh sách file: {e}"