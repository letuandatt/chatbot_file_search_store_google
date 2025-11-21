import base64
import os
from google.genai import types
from chatbot.config import config as app_config
from chatbot.core.utils import image_to_base64
from chatbot.core.history import save_session_message


class VisionService:
    def __init__(self, client):
        self.client = client
        self.model_name = app_config.VISION_MODEL_NAME

    def process_image_query(self, session_id: str, user_id: str, query_text: str, image_path: str):
        """
        Xử lý ảnh, gọi Gemini Vision, và lưu kết quả vào History để Agent có thể nhớ ngữ cảnh sau này (nếu cần).
        """
        if not os.path.exists(image_path):
            return "Lỗi: File ảnh không tồn tại."

        if not self.client:
            return "Lỗi: GenAI Client chưa sẵn sàng."

        image_b64 = image_to_base64(image_path)
        if not image_b64:
            return "Lỗi: Không thể xử lý file ảnh."

        print(f"[VisionService] Processing image for session {session_id}...")

        try:
            # Gọi API (Non-streaming cho đơn giản)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part(text=query_text or "Mô tả bức ảnh này"),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(image_b64)))
                ],
            )

            answer = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))

            # Lưu vào history để đảm bảo tính liên tục của hội thoại
            # Lưu ý: Ta lưu text mô tả, không lưu binary ảnh vào message history text
            save_session_message(
                session_id=session_id,
                user_id=user_id,
                question=f"[Image Query] {query_text} (đã gửi ảnh)",
                answer=answer,
                image_gridfs_id=None  # Có thể update logic lưu ảnh vào GridFS sau
            )

            return answer

        except Exception as e:
            print(f"[VisionService] Error: {e}")
            return "Xin lỗi, tôi gặp sự cố khi phân tích hình ảnh."

