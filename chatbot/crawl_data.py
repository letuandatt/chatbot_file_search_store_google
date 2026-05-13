import os
import re
import time
import requests
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- CẤU HÌNH ---
BASE_OUTPUT_DIR = "data/CongThongTinDienTu/ChiThi"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(BASE_OUTPUT_DIR, "metadata_congthongtindientu_chithi.jsonl")
OS_CHARS_INVALID = r'[<>:"/\\|?*]'  # Các ký tự cấm trong tên file Windows


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    # Thêm user-agent để tránh bị chặn
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def sanitize_filename(name):
    """Làm sạch tên file để lưu được trên ổ cứng"""
    name = re.sub(OS_CHARS_INVALID, "", name)
    name = re.sub(r"\s+", "_", name)  # Thay khoảng trắng bằng _
    return name.strip()[:200]  # Cắt ngắn nếu quá dài


def download_pdf(pdf_url, save_path):
    if os.path.exists(save_path):
        print(f"⚠️ File đã tồn tại: {save_path}")
        return "Đã tồn tại"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Fake header request
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Tải thành công: {os.path.basename(save_path)}")
        return "Tải thành công"
    except Exception as e:
        print(f"❌ Lỗi tải file: {e}")
        return f"Lỗi: {str(e)}"


def append_metadata(metadata):
    with open(METADATA_FILE, "a", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
        f.write("\n")


def load_existing_urls():
    """Dùng URL bài viết làm key check trùng"""
    existing_urls = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        # Lưu URL bài viết gốc để check
                        existing_urls.add(data.get("Nguồn"))
                    except:
                        pass
    return existing_urls


def get_ngay(text):
    """Trích xuất ngày tháng năm"""
    # Tìm chuỗi dạng dd/mm/yyyy
    match = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", text)
    if match:
        return match.group(0)
    return ""


def main():
    driver = setup_driver()
    try:
        existing_urls = load_existing_urls()
        print(f"Đã load {len(existing_urls)} bài viết đã cào trước đó.")

        page_start = 5
        page_end = 8  # Data từ 2020
        for page in range(page_start, page_end + 1):
            print(f"\n>>> 📄 ĐANG CÀO TRANG: {page}")

            # URL tìm kiếm
            url_list = (f"https://congbao.chinhphu.vn/tim-kiem-van-ban?_csrf=WERTZjNNTmINEwoDXwZ9GBUbJjlZBSsRHiExBH8OeCQZNDoyXzktEw%3D%3D&trichyeu="
                        f"&coquanbanhanh=&tungay=01%2F01%2F2020&denngay=&sovanban=&loaivanban=4&nguoiky=&page={page}")

            driver.get(url_list)
            time.sleep(2)  # Chờ JS load list

            soup_list = BeautifulSoup(driver.page_source, "html.parser")
            articles = soup_list.find_all("article", class_="cong-bao-list")

            if not articles:
                print("⚠️ Không tìm thấy bài viết nào. Có thể hết trang hoặc bị chặn.")
                break

            for item in articles:
                try:
                    # 1. Lấy thông tin cơ bản
                    header_tag = item.find("header").find("h1").find("a")
                    ten_van_ban = re.sub(r"\s+", " ", header_tag.get_text(strip=True))
                    real_link = "https://congbao.chinhphu.vn" + header_tag["href"]

                    # CHECK TRÙNG: Nếu link đã cào rồi thì bỏ qua ngay
                    if real_link in existing_urls:
                        print(f"⏭️ Bỏ qua (đã cào): {ten_van_ban[:50]}...")
                        continue

                    print(f"🔍 Đang xử lý: {ten_van_ban}")

                    # Lấy các thông tin khác
                    so_hieu = "Không có số hiệu"
                    match_so = re.search(r"Số:\s*([^\s]+)", ten_van_ban)  # Regex tìm số hiệu tốt hơn
                    if match_so: so_hieu = match_so.group(1)

                    section_p = item.find("section").find("p")
                    summary = section_p.get_text(strip=True) if section_p else ""

                    # Lấy ngày ban hành / hiệu lực
                    ngay_ban_hanh = ""
                    ngay_hieu_luc = ""
                    footer = item.find("footer")
                    if footer:
                        spans = footer.find_all("span")
                        for span in spans:
                            txt = span.get_text(strip=True).lower()
                            if "ban hành" in txt:
                                ngay_ban_hanh = get_ngay(txt)
                            elif "hiệu lực" in txt:
                                ngay_hieu_luc = get_ngay(txt)

                    # 2. Vào trang chi tiết để lấy PDF
                    driver.get(real_link)
                    time.sleep(1.5)  # Chờ load trang con

                    soup_detail = BeautifulSoup(driver.page_source, "html.parser")

                    # Tìm link PDF
                    pdf_url = None
                    pdf_name_raw = f"ChiThi_{so_hieu.replace('/', '-')}.pdf"  # Default name

                    # Cách tìm link PDF của bạn (Dropdown)
                    menu = soup_detail.find("ul", class_="dropdown-menu")
                    if menu:
                        link_tags = menu.find_all("a")
                        for a in link_tags:
                            href = a.get("href", "")
                            if "format=pdf" in href:
                                pdf_url = f"https://congbao.chinhphu.vn{href}"
                                pdf_name_raw = a.get_text(strip=True) or pdf_name_raw
                                break

                    status_tai = "Không tìm thấy link PDF"
                    final_path = ""

                    if pdf_url:
                        # Tạo tên file an toàn
                        safe_name = sanitize_filename(pdf_name_raw)
                        if not safe_name.lower().endswith(".pdf"): safe_name += ".pdf"

                        final_path = os.path.join(BASE_OUTPUT_DIR, safe_name)
                        status_tai = download_pdf(pdf_url, final_path)
                    else:
                        print("⚠️ Không thấy nút tải PDF.")

                    # 3. Lưu Metadata
                    metadata = {
                        "Tên file": final_path,
                        "Tên văn bản": ten_van_ban,
                        "Trích yếu": summary,
                        "Số hiệu": so_hieu,
                        "Ngày ban hành": ngay_ban_hanh,
                        "Ngày hiệu lực": ngay_hieu_luc,
                        "Loại văn bản": "Chỉ thị",
                        "Nguồn": real_link,
                        "PDF URL": pdf_url or "",
                        "Trạng thái tải": status_tai
                    }

                    append_metadata(metadata)
                    existing_urls.add(real_link)  # Cập nhật set để loop sau không bị trùng

                    # Nghỉ nhẹ để tránh DDOS server họ
                    time.sleep(1)

                except Exception as e:
                    print(f"❌ Lỗi xử lý item: {e}")
                    continue

    finally:
        driver.quit()
        print("👋 Đã đóng trình duyệt.")


if __name__ == "__main__":
    main() # Mai check trang Công báo để thu thập toàn bộ file data + update metadata jsonl + điều chỉnh các file liên quan (update profile của chatbot)
