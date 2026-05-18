"""
LexMind chatbot CLI.

Provides the interactive REPL used during development. The heavy
service container is constructed lazily via
`AppContainer.instance()` so importing this module — e.g. from the
FastAPI backend, or from tooling that only wants `handle_pdf_upload`
— does not boot the LLM client or start the file watcher.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage

from chatbot.app_container import AppContainer
from chatbot.core.file_store import save_pdf_to_mongo
from chatbot.core.history import (
    get_session_history,
    list_sessions,
    save_session_message,
)

logger = logging.getLogger(__name__)


def handle_pdf_upload(pdf_path: str, session_id: str, user_id: str) -> None:
    """Store the PDF in MongoDB; the watcher takes over vectorisation."""
    print(f"[main] Đang tải file lên hệ thống: {os.path.basename(pdf_path)}...")

    file_id = save_pdf_to_mongo(pdf_path, session_id, user_id)
    if not file_id:
        print("❌ [main] Lưu file thất bại.")
        return

    print("✅ [main] Đã lưu file. Hệ thống đang xử lý ngầm (Watcher)...")
    # Small delay so the watcher logs an event before the CLI re-prompts.
    time.sleep(1)


def handle_unified_query(
    query_text: str,
    image_path: Optional[str],
    user_id: str,
    session_id: str,
) -> None:
    print("--- Processing by Multi-Agent Graph ---")
    app = AppContainer.instance()

    if app.agent_executor is None:
        print("Agent not ready.")
        return

    try:
        user_profile = app.memory_service.get_profile(user_id) if app.memory_service else None

        inputs = {
            "messages": [HumanMessage(content=query_text)],
            "user_info": user_profile or "Chưa có thông tin.",
            "image_path": image_path,
        }

        result = app.agent_executor.invoke(
            inputs,
            config={"configurable": {"session_id": session_id, "user_id": user_id}},
        )

        last_message = result["messages"][-1]
        full_response = last_message.content
        bot_name = last_message.name if hasattr(last_message, "name") else "Bot"

        print(f"\n🤖 {bot_name}: {full_response}\n")

        save_session_message(
            session_id,
            user_id,
            query_text,
            full_response,
            image_gridfs_id=image_path,
        )
        if app.memory_service is not None:
            app.memory_service.update_profile_background(user_id, query_text)

    except Exception as exc:
        print(f"[main] Agent error: {exc}")


def main() -> None:
    print("🤖 Chatbot Law (Unified Multi-Agent) sẵn sàng!")
    print("=" * 30)

    # Boot the container up front so the first query doesn't pay the
    # GenAI / watcher startup cost interactively.
    AppContainer.instance()

    # Mock User ID — for production the API supplies user_id from auth.
    user_id = "6935267b0d228c9dbb5d0ecc"

    print("[1] Tạo session mới")
    print("[2] Tiếp tục session cũ")
    choice = input("Lựa chọn (1/2): ").strip()

    if choice == "2":
        sessions = list_sessions(limit=10, user_id=user_id)
        if not sessions:
            session_id = str(uuid.uuid4())
        else:
            for i, s in enumerate(sessions):
                print(f"  [{i + 1}] {s['session_id']} ({s['num_messages']} msgs)")
            try:
                s_choice = int(input("Chọn (0=Mới): ").strip())
                if 0 < s_choice <= len(sessions):
                    session_id = sessions[s_choice - 1]["session_id"]
                else:
                    session_id = str(uuid.uuid4())
            except ValueError:
                session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())

    print(f"\n🆔 Session ID: {session_id}")
    print("Gõ 'pdf' để tải file, 'exit' để thoát.\n")

    get_session_history(session_id, user_id)

    while True:
        user_input = input("\n👤 Bạn: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "pdf":
            path = input("📂 PDF Path: ").strip().replace('"', "")
            if os.path.exists(path):
                handle_pdf_upload(path, session_id, user_id)
            else:
                print("File không tồn tại.")
            continue

        img_path: Optional[str] = input("🖼️ Ảnh Path (Enter để bỏ qua): ").strip().replace('"', "")
        if not img_path:
            img_path = None
        elif not os.path.exists(img_path):
            print("⚠️ File ảnh không tồn tại. Tiếp tục chỉ với text.")
            img_path = None

        handle_unified_query(user_input, img_path, user_id, session_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as exc:
        print(f"[main] Fatal Error: {exc}")
    finally:
        # Cleanly stop the file watcher so we don't leave a daemon thread
        # running after the REPL exits.
        try:
            AppContainer.instance().shutdown()
        except Exception:
            pass
