from langchain_core.tools import tool
from chatbot.core.file_store import get_session_file_stores
from chatbot.core.utils import safe_json_parse, get_cache, set_cache, cache_key, rerank_text_snippet, extract_citations
from chatbot.config import config as app_config
import google.genai.types as types

# will be set by main
GLOBAL_GENAI_CLIENT = None

def set_global_genai(client):
    global GLOBAL_GENAI_CLIENT
    GLOBAL_GENAI_CLIENT = client

def _clean_stores(stores: list[str]):
    """
    CHỈ dùng để tìm kiếm trong các FILE DO NGƯỜI DÙNG TẢI LÊN trong phiên làm việc hiện tại.
    Sử dụng tool này khi câu hỏi liên quan đến dữ liệu riêng tư, báo cáo cụ thể, hoặc user nói "trong file tôi vừa gửi", "tóm tắt file này".
    NẾU câu hỏi mang tính chất quy định chung, quy trình chuẩn của tổ chức -> HÃY DÙNG tool_search_general_policy thay thế.
    """
    if not stores:
        return []
    cleaned = [s for s in stores if s and isinstance(s, str) and s.strip()]
    if GLOBAL_GENAI_CLIENT is None:
        return cleaned
    valid = []
    for s in cleaned:
        try:
            GLOBAL_GENAI_CLIENT.file_search_stores.get(name=s)
            valid.append(s)
        except Exception:
            # ignore invalid store
            continue
    return valid

@tool
def tool_search_uploaded_file(query: str = None, session_id: str = None, **kwargs):
    """
    Search inside user's uploaded files for the given session's file stores.
    Input flexible: query can be a JSON-like string containing {"query": "...", "session_id":"..."}.
    Returns short summary + citations.
    """
    # normalize input
    q_in = query if query is not None else kwargs.get("query")
    parsed = safe_json_parse(q_in)
    if isinstance(parsed, dict) and parsed.get("query"):
        q_in = parsed.get("query")
        if parsed.get("session_id"):
            session_id = parsed.get("session_id")

    if not q_in or not str(q_in).strip():
        return "Lỗi: Không có 'query' để tìm kiếm trong file."

    if not session_id:
        return "Lỗi: Không có 'session_id' (không biết chọn store)."

    query_text = str(q_in).strip()
    # get stores for session
    user_file_stores = get_session_file_stores(session_id)
    user_file_stores = _clean_stores(user_file_stores)
    if not user_file_stores:
        return "Phiên này chưa có tài liệu đã được xử lý (processed)."

    cache_k = cache_key("file", session_id, query_text)
    cached = get_cache(cache_k)
    if cached:
        return cached

    if GLOBAL_GENAI_CLIENT is None:
        return "Lỗi: Google GenAI client chưa khởi tạo."

    try:
        response = GLOBAL_GENAI_CLIENT.models.generate_content(
            model=app_config.TEXT_MODEL_NAME,
            contents=[types.Part(text=query_text)],
            config=types.GenerateContentConfig(
                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=user_file_stores))]
            ),
        )
    except Exception as e:
        return f"Lỗi khi tra cứu file tải lên: {e}"

    try:
        text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
    except Exception:
        text = None

    if not text:
        out = "Không tìm thấy thông tin trong tài liệu đã tải lên."
        set_cache(cache_k, out)
        return out

    text_rerank = rerank_text_snippet(text, max_lines=12)
    citations = extract_citations(response)
    out = (text_rerank or text) + citations
    set_cache(cache_k, out)
    return out