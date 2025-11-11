import google.genai as genai
import google.genai.types as types
import sys

from chatbot import config

# --- Hằng số từ Config ---
GOOGLE_API_KEY = config.GOOGLE_API_KEY
STORE_NAME = config.CUSC_MAIN_STORE_NAME


def test_query(client, store_name, test_question):
    """
    Gửi một câu hỏi test đến File Store.
    """
    print(f"🚀 Đang test query với Store: {store_name}")
    print(f"❓ Câu hỏi: {test_question}\n")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=test_question,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ]
            ),
        )

        print("✅ TRẢ LỜI TỪ RAG:\n")
        print(response.text)
        print()

        # --- Nguồn trích dẫn ---
        metadata = response.candidates[0].grounding_metadata

        # Kiểm tra xem có metadata, support, và chunk không
        if not (metadata and metadata.grounding_supports and metadata.grounding_chunks):
            print("(Không tìm thấy thông tin trích dẫn chi tiết)")
            return  # Kết thúc hàm sớm

        # 1. Lấy danh sách TẤT CẢ chunk (để tra cứu tên file)
        all_chunks = metadata.grounding_chunks

        # 2. Tạo một dictionary để nhóm các trích dẫn theo tên file
        citations_by_file = {}

        # 3. Lặp qua các 'grounding_supports' (đây là các trích dẫn thực tế)
        for support in metadata.grounding_supports:

            # Lấy đoạn văn bản chính xác đã được AI sử dụng
            segment_text = support.segment.text

            # Lấy các chunk (file) mà đoạn văn bản này thuộc về
            for chunk_index in support.grounding_chunk_indices:
                if 0 <= chunk_index < len(all_chunks):
                    chunk = all_chunks[chunk_index]
                    filename = chunk.retrieved_context.title

                    # Thêm vào dictionary
                    if filename not in citations_by_file:
                        citations_by_file[filename] = set()  # Dùng set để tránh trùng lặp

                    citations_by_file[filename].add(segment_text)

        # 4. In ra kết quả đã được nhóm lại
        if not citations_by_file:
            print("(Không tìm thấy trích dẫn cụ thể)")
        else:
            for filename, segments in citations_by_file.items():
                print(f"Nguồn: {filename}")
                print("-" * 20)

    except Exception as e:
        print(f"❌ Lỗi khi thực hiện test query: {e}")


# ==============================================================================
# MAIN LOGIC
# ==============================================================================
if __name__ == '__main__':
    # 1. Kiểm tra xem file .env đã được cập nhật chưa
    if not STORE_NAME:
        print("❌ LỖI: CUSC_MAIN_STORE_NAME bị trống trong file .env.")
        print("Vui lòng chạy 'python setup_main_store.py' trước và cập nhật file .env.")
        sys.exit()  # Thoát

    # 2. Khởi tạo client
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tạo client: {e}")
        sys.exit()

    # 3. Lấy câu hỏi
    test_question = input("Nhập câu hỏi test (Enter để dùng câu mặc định): ")
    if not test_question.strip():
        test_question = "Mục đích của thủ tục kiểm định TT07.05.I là gì?"
        print(f"Sử dụng câu hỏi mặc định: {test_question}")

    # 4. Chạy test
    test_query(client, STORE_NAME, test_question)