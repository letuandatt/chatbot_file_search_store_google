from langchain_core.tools import tool
from chatbot.core.db import DB_COLLECTION


@tool
def tool_recall_chat_history(user_id: str, session_id: str):
    """
    Công cụ truy xuất lịch sử chat của MỘT PHIÊN CỤ THỂ (Session).
    Dùng khi người dùng hỏi: "Nãy giờ tôi hỏi gì?", "Tóm tắt cuộc trò chuyện này", "Tôi đã hỏi gì trong phiên này".

    Args:
        user_id: ID người dùng (Lấy từ ngữ cảnh).
        session_id: ID của phiên làm việc cần tra cứu (Lấy từ ngữ cảnh).
    """
    if not user_id or not session_id:
        return "Lỗi: Thiếu user_id hoặc session_id."

    if DB_COLLECTION is None:
        return "Lỗi: DB chưa kết nối."

    try:
        # --- QUERY CHÍNH XÁC 1 SESSION ---
        session = DB_COLLECTION.find_one({
            "user_id": user_id,
            "session_id": session_id
        })

        if not session:
            return f"Không tìm thấy dữ liệu cho phiên làm việc {session_id}."

        messages = session.get("messages", [])

        # Đếm và lọc tin nhắn
        question_count = sum(1 for m in messages if m.get("question"))

        if question_count == 0:
            return "Phiên này chưa có câu hỏi nào."

        summary = f"Trong phiên làm việc {session_id}, bạn đã hỏi {question_count} câu:\n"

        for m in messages:
            q = m.get("question", "")
            if q:
                summary += f"- {q}\n"

        return summary

    except Exception as e:
        return f"Lỗi truy xuất lịch sử: {e}"
