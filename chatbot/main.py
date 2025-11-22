import os
import uuid
import google.genai as genai
from bson.objectid import ObjectId
from chatbot.config import config as app_config

# Core Imports
from chatbot.core.db import init_db, DB_DOCUMENTS_COLLECTION
from chatbot.core.history import list_sessions, get_session_history, save_session_message
from chatbot.core.file_store import save_pdf_to_mongo, process_and_vectorize_pdf

# Services & Router
from chatbot.services.vision_service import VisionService
from chatbot.router.dispatcher import build_rag_agent

from langchain_core.messages import HumanMessage


# --- SERVICE CONTAINER ---
class AppContainer:
    """
    Quản lý khởi tạo Client và các Service (Singleton-like)
    """

    def __init__(self):
        init_db()
        try:
            self.genai_client = genai.Client(api_key=app_config.GOOGLE_API_KEY)
            print("[App] GenAI Client Initialized.")
        except Exception as e:
            print(f"[App] GenAI Client Init Failed: {e}")
            self.genai_client = None

        # Init Vision Service
        self.vision_service = VisionService(self.genai_client)

        # Init Agent
        if self.genai_client:
            self.agent_executor, self.text_llm = build_rag_agent(self.genai_client)  #
        else:
            self.agent_executor = None


# Khởi tạo App toàn cục
APP = AppContainer()


# --- HELPER FUNCTIONS ---
def handle_pdf_upload(pdf_path: str, session_id: str, user_id: str):
    print(f"[main] Uploading file for session {session_id} ...")
    file_id = save_pdf_to_mongo(pdf_path, session_id, user_id)  #
    if not file_id:
        print("[main] save failed.")
        return

    # Check status
    try:
        doc = DB_DOCUMENTS_COLLECTION.find_one({"_id": ObjectId(file_id)})
    except Exception:
        doc = None

    if doc and doc.get("status") == "processed":
        print("[main] File already processed.")
    else:
        # Sử dụng Client từ APP Container
        process_and_vectorize_pdf(pdf_path, session_id, str(doc["_id"]), APP.genai_client)  #
        print("[main] Processed and created file store.")


def handle_text_query(query_text: str, user_id: str, session_id: str):
    print("--- Processing by Multi-Agent System ---")
    if not APP.agent_executor:
        print("Agent not ready.")
        return
    try:
        # LangGraph input là một list messages
        inputs = {"messages": [HumanMessage(content=query_text)]}

        # Gọi Graph
        # config dùng để quản lý state nếu cần (nhưng ở đây state lưu trong graph memory tạm)
        result = APP.agent_executor.invoke(inputs, config={"configurable": {"session_id": session_id, "user_id": user_id}})

        # Lấy tin nhắn cuối cùng của AI
        last_message = result["messages"][-1]
        full_response = last_message.content

        print(f"\n🤖 Bot ({last_message.name if hasattr(last_message, 'name') else 'Assistant'}): {full_response}\n")

        save_session_message(session_id, user_id, query_text, full_response)
    except Exception as e:
        print(f"[main] Agent error: {e}")


# --- MAIN FUNCTION (UPDATED) ---
def main():
    print("🤖 Chatbot CUSC (Agent + Google File Search) sẵn sàng!")
    print("=" * 30)
    print("[1] Tạo session mới")
    print("[2] Tiếp tục session cũ")

    user_id = "6915f6a4d74b46caa1d4d0b2"
    choice = input("Lựa chọn của bạn (1 hoặc 2): ").strip()

    if choice == '2':
        sessions = list_sessions(limit=10, user_id=user_id)  #
        if not sessions:
            session_id = str(uuid.uuid4())
        else:
            for i, s in enumerate(sessions):
                print(f"  [{i + 1}] {s['session_id']} ({s['num_messages']} tin nhắn, cập nhật: {s['updated_at']})")
            try:
                s_choice = int(input("Chọn session (0 để tạo mới): ").strip())
                if 0 < s_choice <= len(sessions):
                    session_id = sessions[s_choice - 1]['session_id']
                else:
                    session_id = str(uuid.uuid4())
            except:
                session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())

    print(f"\n🆔 Session ID: {session_id}")
    print("Gõ 'pdf' để tải file, 'exit' để thoát.\n")

    get_session_history(session_id, user_id)  # Pre-load history

    while True:
        user_input = input("\n👤 Bạn: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "pdf":
            path = input("📂 PDF Path: ").strip().replace('"', '')
            if os.path.exists(path):
                handle_pdf_upload(path, session_id, user_id)
            else:
                print("File không tồn tại.")
            continue

        img_path = input("🖼️ Ảnh Path (Enter để bỏ qua): ").strip().replace('"', '')
        if img_path and os.path.exists(img_path):
            # [Refactor] Sử dụng VisionService từ APP Container thay vì hàm rời rạc cũ
            vision_resp = APP.vision_service.process_image_query(session_id, user_id, user_input, img_path)
            print(f"\n🤖 Vision: {vision_resp}\n")
        else:
            handle_text_query(user_input, user_id, session_id)


if __name__ == "__main__":
    main()
