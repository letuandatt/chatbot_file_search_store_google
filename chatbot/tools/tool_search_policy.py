from langchain_core.tools import tool
from chatbot.core.utils import get_cache, set_cache, cache_key, rerank_text_snippet, extract_citations
from chatbot.config import config as app_config
import google.genai.types as types

# will be set by main
GLOBAL_GENAI_CLIENT = None

def set_global_genai(client):
    global GLOBAL_GENAI_CLIENT
    GLOBAL_GENAI_CLIENT = client

@tool
def tool_search_general_policy(query: str):
    """
    Dùng để tra cứu các QUY ĐỊNH CHUNG, QUY TRÌNH, THỦ TỤC, BIỂU MẪU, HƯỚNG DẪN của tổ chức (CUSC).
    Sử dụng tool này khi câu hỏi chứa các từ khóa: "quy trình", "thủ tục", "quy định", "ISO", "hướng dẫn", "biểu mẫu", "TT" (thông tư/thủ tục).
    Ví dụ: "Quy trình kiểm định", "Quy định nghỉ phép", "Biểu mẫu báo cáo", "Thủ tục TT07".
    KHÔNG dùng tool này để tìm kiếm thông tin trong file người dùng tự upload (CV, báo cáo cá nhân...).
    """
    if not app_config.CUSC_MAIN_STORE_NAME:
        return "Hệ thống chưa được cấu hình Main Store."
    if not query or not str(query).strip():
        return "Lỗi: truy vấn rỗng."

    cache_k = cache_key("policy", "main", query)
    cached = get_cache(cache_k)
    if cached:
        return cached

    if GLOBAL_GENAI_CLIENT is None:
        return "Lỗi: Google GenAI client chưa khởi tạo."

    try:
        response = GLOBAL_GENAI_CLIENT.models.generate_content(
            model=app_config.TEXT_MODEL_NAME,
            contents=[types.Part(text=query)],
            config=types.GenerateContentConfig(
                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[app_config.CUSC_MAIN_STORE_NAME]))]
            ),
        )
        text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
        text = rerank_text_snippet(text, max_lines=12)
        citations = extract_citations(response)
        out = (text or "Không tìm thấy thông tin trong Main Store.") + citations
        set_cache(cache_k, out)
        return out
    except Exception as e:
        return f"Lỗi khi tra cứu quy trình chung: {e}"