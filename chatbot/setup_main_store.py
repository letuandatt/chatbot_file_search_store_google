# Tên file: setup_main_store.py (ĐÃ SỬA)
import google.genai as genai
import google.genai.types as types
import os
import time

# Import config ĐÃ SỬA
from chatbot import config

# --- Hằng số từ Config ---
GOOGLE_API_KEY = config.GOOGLE_API_KEY
DATA_DIR = config.DATA_DIR


# (Xóa biến STORE_NAME toàn cục không cần thiết)


def create_and_populate_store(client):
    """
    Tạo File Store, tải file lên, và chờ index.
    """
    print("Đang tạo File Store mới trên Google...")
    try:
        # === SỬA LỖI LOGIC ===
        # display_name là một tên cố định, không phải biến
        file_store = client.file_search_stores.create(
            config={'display_name': 'CUSC Main Knowledge'}
        )
        store_name = file_store.name

        # In ra hướng dẫn CỰC KỲ QUAN TRỌNG
        print("\n" + "=" * 50)
        print(f"✅ TẠO THÀNH CÔNG STORE: {store_name}")
        print("BÂY GIỜ HÃY MỞ FILE .env VÀ DÁN DÒNG SAU VÀO:")
        print(f'CUSC_MAIN_STORE_NAME="{store_name}"')
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"Lỗi khi tạo File Store: {e}")
        return None

    # 3. Tải các file lên (Code này đã đúng)
    print(f"Bắt đầu tải file từ thư mục: {DATA_DIR}...")
    file_count = 0
    files_to_upload = []
    last_filename = None

    for filename in os.listdir(DATA_DIR):
        if os.path.isdir(os.path.join(DATA_DIR, filename)): continue
        if filename.endswith((".pdf", ".md", ".txt", ".docx")):
            files_to_upload.append(filename)

    total_files = len(files_to_upload)
    if total_files == 0:
        print("Cảnh báo: Không tìm thấy file nào trong DATA_DIR.")
        return store_name

    print(f"Tìm thấy {total_files} file để tải lên.")

    for i, filename in enumerate(files_to_upload):
        file_path = os.path.join(DATA_DIR, filename)
        last_filename = filename
        try:
            print(f"Đang tải lên file {i + 1}/{total_files}: {filename}...")
            client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=store_name,
                config={'display_name': filename}
            )
            file_count += 1
        except Exception as e:
            print(f"Lỗi khi tải file {filename}: {e}")

    print(f"\nHoàn tất! Đã tải {file_count} / {total_files} file lên Store.")

    print("\n✅ QUY TRÌNH SETUP HOÀN TẤT.")
    return store_name


# ==============================================================================
# MAIN LOGIC (Chỉ để chạy hàm setup)
# ==============================================================================
if __name__ == '__main__':
    print("Đang khởi tạo Google Client...")
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tạo client: {e}")
        exit()

    # Chạy hàm tạo mới
    create_and_populate_store(client)
    print("\nLƯU Ý: Hãy cập nhật file .env trước khi chạy test_query.py.")