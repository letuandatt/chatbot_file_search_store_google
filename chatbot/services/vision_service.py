import base64
import os
from google.genai import types
from chatbot.config import config as app_config
from chatbot.core.utils import image_to_base64

class VisionService:
    def __init__(self, client):
        self.client = client
        self.model_name = app_config.VISION_MODEL_NAME

    def analyze_image(self, query_text: str, image_path: str) -> str:
        """
        Phân tích ảnh và trả về nội dung mô tả/trả lời.
        Pure function: Không lưu DB, không phụ thuộc session_id.
        """
        if not image_path or not os.path.exists(image_path):
            return "Lỗi hệ thống: File ảnh không tồn tại hoặc đường dẫn sai."

        if not self.client:
            return "Lỗi hệ thống: GenAI Client chưa sẵn sàng."

        image_b64 = image_to_base64(image_path)
        if not image_b64:
            return "Lỗi hệ thống: Không thể mã hóa file ảnh."

        print(f"[VisionService] Analyzing image: {image_path}...")

        try:
            # Nếu user không hỏi gì, mặc định là yêu cầu mô tả
            final_prompt = query_text if query_text and query_text.strip() else "Mô tả chi tiết nội dung trong bức ảnh này."

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part(text=final_prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(image_b64)))
                ],
            )

            answer = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
            return answer

        except Exception as e:
            print(f"[VisionService] Error: {e}")
            return f"Xin lỗi, tôi gặp sự cố khi phân tích hình ảnh: {str(e)}"
