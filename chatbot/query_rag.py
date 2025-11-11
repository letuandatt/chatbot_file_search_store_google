import os
import io
import base64
import uuid
import gridfs
import pytz
import hashlib
import google.genai as genai
import google.genai.types as types
from chatbot import config as app_config

from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from PIL import Image

# --- LangChain Imports ---
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# ==============================================================================
# SECTION 1: KHỞI TẠO CÁC THÀNH PHẦN TOÀN CỤC (GLOBAL COMPONENTS)
# ==============================================================================

# --- MONGODB CONNECTION ---
try:
    _mongo_client = MongoClient(app_config.MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    _mongo_client.admin.command('ping')
    print("MongoDB ping successful.")

    _mongo_db = _mongo_client[app_config.MONGO_DB_NAME]
    DB_COLLECTION = _mongo_db["sessions"]
    FS = gridfs.GridFS(_mongo_db)

    DB_COLLECTION.create_index([("session_id", ASCENDING)], unique=True)
    DB_COLLECTION.create_index([("updated_at", DESCENDING)])
    print(f"Connected successfully to MongoDB and GridFS.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    DB_COLLECTION, FS, _mongo_db = None, None, None

# --- GOOGLE AI SDK CLIENT (MỚI) ---
try:
    GLOBAL_GENAI_CLIENT = genai.Client()
    print("Google Generative AI SDK client initialized.")
except Exception as e:
    print(f"Lỗi khi cấu hình Google AI SDK client: {e}")
    GLOBAL_GENAI_CLIENT = None


def get_mongo_collection(collection_name: str = "sessions"):
    """Trả về collection 'sessions' đã được khởi tạo."""
    if _mongo_client is None or _mongo_db is None:
        print(f"Lỗi: Kết nối MongoDB chưa được thiết lập")
        return None
    try:
        return _mongo_db[collection_name]
    except Exception as ex:
        print(f"Lỗi khi lấy collection '{collection_name}': {ex}")
        return None


try:
    DB_DOCUMENTS_COLLECTION = get_mongo_collection("documents")
    if DB_DOCUMENTS_COLLECTION is not None:
        DB_DOCUMENTS_COLLECTION.create_index([("session_id", ASCENDING)])
        DB_DOCUMENTS_COLLECTION.create_index([("user_id", ASCENDING)])
        DB_DOCUMENTS_COLLECTION.create_index([("created_at", DESCENDING)])
        print("MongoDB collection 'documents' initialized.")
except Exception as e:
    print(f"Failed to initialize 'documents' collection: {e}")
    DB_DOCUMENTS_COLLECTION = None


def check_session_belongs_to_user(session_id: str, user_id: str) -> bool:
    """Kiểm tra session có tồn tại và thuộc về user_id không."""
    coll = get_mongo_collection("sessions")  # Lấy collection sessions
    if coll is None:
        return False
    try:
        # Đếm số document khớp cả session_id và user_id
        return coll.count_documents({"session_id": session_id, "user_id": user_id}, limit=1) > 0
    except Exception as e:
        print(f"Lỗi khi kiểm tra session ownership: {e}")
        return False


# --- VIETNAM TIMEZONE DEFINITION (Giữ nguyên) ---
try:
    VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
    print("VN_TZ initialized successfully.")
except pytz.UnknownTimeZoneError:
    VN_TZ = timezone.utc


# --- LLM MODEL (Giữ nguyên) ---
def initialize_llm(model_name, temperature):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )


try:
    TEXT_LLM = initialize_llm(app_config.TEXT_MODEL_NAME, 0.1)
    VISION_LLM = initialize_llm(app_config.VISION_MODEL_NAME, 0.1)
    print("LLM (LangChain) models initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize LLMs: {e}")
    TEXT_LLM, VISION_LLM = None, None


# ==============================================================================
# SECTION 2: CÁC HÀM TIỆN ÍCH CỐT LÕI (CORE UTILITY FUNCTIONS)
# ==============================================================================
def format_chat_history(history):
    """Format lịch sử chat thành chuỗi văn bản để đưa vào prompt."""
    if not history:
        return "No previous messages."
    formatted_parts = []
    for message in history:
        role = getattr(message, 'role', str(type(message).__name__))
        content = getattr(message, 'content', str(message))
        if isinstance(content, list):
            # Xử lý nội dung multimodal (chỉ lấy text)
            text_content = ""
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content = part.get("text", "")
                    break
            content = text_content
        formatted_parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(formatted_parts)


def extract_citations(response):
    """Trích xuất và format nguồn trích dẫn từ response metadata, giống test_query.py."""
    citations_str = "\n\n--- Nguồn trích dẫn ---\n"
    try:
        metadata = response.candidates[0].grounding_metadata
        if not (metadata and metadata.grounding_supports and metadata.grounding_chunks):
            return ""  # Không có trích dẫn

        all_chunks = metadata.grounding_chunks
        citations_by_file = {}

        for support in metadata.grounding_supports:
            segment_text = support.segment.text
            for chunk_index in support.grounding_chunk_indices:
                if 0 <= chunk_index < len(all_chunks):
                    chunk = all_chunks[chunk_index]
                    filename = chunk.retrieved_context.title
                    if filename not in citations_by_file:
                        citations_by_file[filename] = set()
                    citations_by_file[filename].add(segment_text)

        if not citations_by_file:
            return ""

        for filename, segments in citations_by_file.items():
            citations_str += f"Nguồn: {filename}\n"
            citations_str += "-" * 20 + "\n"
            for segment in segments:
                citations_str += f"{segment}\n"
            citations_str += "\n"

        return citations_str
    except Exception as e:
        print(f"Lỗi khi trích xuất citations: {e}")
        return ""


def save_session_message(session_id, user_id, question, answer, image_path=None):
    """Lưu câu hỏi và câu trả lời vào MongoDB (bản tối ưu)."""
    coll = get_mongo_collection()
    fs_client = FS
    if coll is None or fs_client is None:
        print("Lỗi: Không thể lưu session, DB hoặc GridFS chưa kết nối.")
        return

    now = datetime.now(VN_TZ).isoformat()

    image_gridfs_id = None

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as i_f:
                image_gridfs_id = fs_client.put(
                    i_f,
                    filename=os.path.basename(image_path),
                    metadata={
                        "session_id": session_id,
                        "created_at": now,
                        "updated_at": now
                    }
                )
        except Exception as ex:
            print(f"Lỗi khi lưu ảnh vào GridFS: {ex}")

    new_messages = [
        {
            "role": "user",
            "content": question,
            "image_gridfs_id": str(image_gridfs_id) if image_gridfs_id else None,
            "timestamp": now
        },
        {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now(VN_TZ).isoformat()
        }
    ]

    coll.update_one(
        {"session_id": session_id, "user_id": user_id},
        {
            "$push": {"messages": {"$each": new_messages}},
            "$set": {"updated_at": datetime.now(VN_TZ).isoformat()},
            "$setOnInsert": {  # <-- Chỉ set các trường này khi TẠO MỚI
                "created_at": now
            }
        },
        upsert=True  # <-- Tự động tạo nếu chưa có
    )


def load_session_messages(session_id: str, user_id: str, max_history_message: int = 50):
    """Load lịch sử hội thoại từ MongoDB."""
    coll = get_mongo_collection("sessions")
    fs_client = FS
    if coll is None or fs_client is None:
        return InMemoryChatMessageHistory()

    history = InMemoryChatMessageHistory()

    try:
        session_doc = coll.find_one(
            {"session_id": session_id, "user_id": user_id},
            projection={"messages": {"$slice": -max_history_message}}
        )

        if not session_doc:
            print(f"DEBUG: Session {session_id} not found or doesn't belong to user {user_id}")
            return history

        for msg in session_doc.get("messages", []):
            if msg["role"] == "user":
                image_gridfs_id_str = msg.get("image_gridfs_id")
                content_list = [{"type": "text", "text": msg["content"]}]
                if image_gridfs_id_str:
                    try:
                        image_id = ObjectId(image_gridfs_id_str)
                        image_data = fs_client.get(image_id)  # Dùng fs_client
                        image_base64 = base64.b64encode(image_data.read()).decode("utf-8")
                        content_list.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}} )
                    except Exception as ex:
                        print(f"Lỗi khi tải ảnh từ GridFS (ID: {image_gridfs_id_str}): {ex}")
                history.add_message(HumanMessage(content=content_list))
            elif msg["role"] == "assistant":
                history.add_message(AIMessage(content=msg["content"]))
            else:
                print(f"⚠️ Unknown role: {msg['role']}")
    except Exception as e:
        print(f"Lỗi khi tải session ({session_id}) từ MongoDB: {e}")
        # Trả về history rỗng để tránh crash
        return InMemoryChatMessageHistory()

    return history


def list_sessions(user_id: str, limit=50):
    """Liệt kê các session (đã tối ưu) mà không tải messages."""
    coll = get_mongo_collection("sessions")
    if coll is None:
        return []

    pipeline = [
        {
            "$match": {"user_id": user_id}
        },
        {
            "$project": {  # Chỉ lấy các trường này
                "_id": 0,
                "session_id": 1,
                "session_name": 1,
                "updated_at": 1,
                "created_at": 1,
                "num_messages": {"$size": "$messages"}  # Yêu cầu DB đếm
            }
        },
        {
            "$sort": {"updated_at": DESCENDING}
        },
        {
            "$limit": limit  # Chỉ lấy 50 session gần nhất
        }
    ]

    try:
        sessions = list(coll.aggregate(pipeline))
        return sessions
    except Exception as e:
        print(f"Lỗi khi list sessions: {e}")
        return []

# --- (Các hàm quản lý file giữ nguyên, chúng ĐÃ ĐÚNG) ---
def get_session_file_store(session_id: str) -> str | None:
    # (Giữ nguyên)
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return None
    try:
        doc_record = coll.find_one(
            {"session_id": session_id, "status": "processed"},
            projection={"file_store_name": 1}
        )
        return doc_record.get("file_store_name") if doc_record else None
    except Exception:
        return None


def compute_file_hash(file_path: str) -> str:
    """Tạo hash MD5 cho file để tránh trùng."""
    with open(file_path, "rb") as f:
        file_data = f.read()
    return hashlib.md5(file_data).hexdigest()


def save_pdf_to_mongo(file_path: str, session_id: str, user_id: str) -> str | None:
    fs_client = FS
    coll = DB_DOCUMENTS_COLLECTION
    if fs_client is None or coll is None:
        print("Lỗi: Không thể lưu file, DB hoặc GridFS chưa kết nối.")
        return None

    now = datetime.now(VN_TZ).isoformat()
    file_name = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)  # ✅ thêm dòng này

    try:
        with open(file_path, "rb") as f:
            file_id = fs_client.put(
                f,
                filename=file_name,
                metadata={
                    "session_id": session_id,
                    "user_id": user_id,
                    "file_hash": file_hash,
                    "created_at": now
                }
            )

        doc_record = {
            "session_id": session_id,
            "user_id": user_id,
            "filename": file_name,
            "gridfs_id": str(file_id),
            "file_hash": file_hash,  # ✅ thêm vào đây
            "created_at": now,
            "status": "uploaded"
        }
        coll.insert_one(doc_record)
        print(f"Đã lưu file '{file_name}' vào GridFS (ID: {file_id}) và collection 'documents'.")
        return str(file_id)
    except Exception as e:
        print(f"Lỗi khi lưu file PDF vào MongoDB: {e}")
        return None


def process_and_vectorize_pdf(file_path: str, session_id: str, user_id: str):
    """

    :param file_path:
    :param session_id:
    :param user_id:
    :return:
    """
    if DB_DOCUMENTS_COLLECTION is None or GLOBAL_GENAI_CLIENT is None:
        return

    client = GLOBAL_GENAI_CLIENT
    file_name = os.path.basename(file_path)
    print(f"Bắt đầu xử lý và tải file lên Google: {file_name}")
    try:
        print(f"Đang tạo File Store mới cho session {session_id}...")
        file_store = client.file_search_stores.create(
            config={'display_name': f"Session Store - {session_id} - {file_name}"}
        )
        store_name = file_store.name
        print(f"Tạo thành công File Store: {store_name}")

        print(f"Đang tải file {file_name} lên store...")
        client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config={'display_name': file_name}
        )
        print("Tải file lên thành công.")

        DB_DOCUMENTS_COLLECTION.update_one(
            {"session_id": session_id, "filename": file_name, "user_id": user_id},
            {"$set": {"status": "processed", "file_store_name": store_name}}
        )
        print(f"Đã cập nhật MongoDB, liên kết session với store: {store_name}")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi xử lý file với Google File Search: {e}")
        DB_DOCUMENTS_COLLECTION.update_one(
            {"session_id": session_id, "filename": file_name, "user_id": user_id},
            {"$set": {"status": "error_processing"}}
        )


def delete_session_and_associated_files(session_id: str, user_id: str) -> dict:
    """

    :param session_id:
    :param user_id:
    :return:
    """
    sessions_coll = get_mongo_collection("sessions")
    docs_coll = DB_DOCUMENTS_COLLECTION
    fs_client = FS
    client = GLOBAL_GENAI_CLIENT
    if sessions_coll is None or fs_client is None or client is None or docs_coll is None:
        raise Exception("Một hoặc nhiều thành phần DB (Mongo, GridFS, client) chưa được khởi tạo")

    deleted_counts = {
        "sessions": 0,
        "document_records": 0,
        "gridfs_files": 0,
        "file_stores": 0
    }

    gridfs_ids_to_delete, file_store_names_to_delete = [], set()

    try:
        session_doc = sessions_coll.find_one({"session_id": session_id, "user_id": user_id})
        if session_doc:
            for msg in session_doc.get("messages", []):
                if msg.get("image_gridfs_id"):
                    try:
                        gridfs_ids_to_delete.append(ObjectId(msg["image_gridfs_id"]))
                    except Exception:
                        pass
        doc_records = list(docs_coll.find({"session_id": session_id, "user_id": user_id}))
        for doc in doc_records:
            if doc.get("gridfs_id"):
                try:
                    gridfs_ids_to_delete.append(ObjectId(doc["gridfs_id"]))
                except Exception:
                    pass
            if doc.get("file_store_name"):
                file_store_names_to_delete.add(doc["file_store_name"])
    except Exception as e:
        print(f"Lỗi khi thu thập ID: {e}")

    for file_id in set(gridfs_ids_to_delete):
        try:
            fs_client.delete(file_id); deleted_counts["gridfs_files"] += 1
        except Exception:
            pass

    deleted_counts["sessions"] = sessions_coll.delete_one({"session_id": session_id, "user_id": user_id}).deleted_count
    deleted_counts["document_records"] = docs_coll.delete_many({"session_id": session_id, "user_id": user_id}).deleted_count

    for store_name in file_store_names_to_delete:
        try:
            print(f"Đang xóa File Store: {store_name}...")
            client.file_search_stores.delete(name=store_name)
            deleted_counts["file_stores"] += 1
            print(f"Đã xóa {store_name}.")
        except Exception as e:
            print(f"Lỗi khi xóa File Store {store_name}: {e}")
    return deleted_counts


def image_to_base64(image_path, max_size_px=1024, jpeg_quality=85):
    """Chuyển file ảnh sang chuỗi base64, đồng thời
    resize và nén ảnh để tối ưu chi phí và tốc độ.
    """
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size_px, max_size_px))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(
                buffered,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True
            )
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return None


# --- MEMORY MANAGEMENT ---
def get_session_history(session_id: str, user_id: str):
    """Lấy lịch sử chat TRỰC TIẾP từ MongoDB cho user cụ thể."""
    print(f"--- DEBUG: Loading history for session '{session_id}' / user '{user_id}' from DB ---")
    return load_session_messages(session_id, user_id)


# ==============================================================================
# SECTION 3: CÁC HÀM TẠO CHAIN (CHAIN FACTORY FUNCTIONS)
# ==============================================================================

# --- PROMPTS (Cập nhật với format) ---
ROUTER_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Bạn là AI phân loại câu hỏi. Dựa trên Lịch sử trò chuyện và Câu hỏi mới,
hãy phân loại câu hỏi vào MỘT trong ba loại sau:
1.  `rag_query`: Câu hỏi yêu cầu thông tin về quy trình, thủ tục, hoặc thông tin chung.
2.  `history_query`: Câu hỏi về chính cuộc hội thoại.
3.  `file_rag_query`: Câu hỏi liên quan đến tài liệu, file (PDF) MÀ NGƯỜI DÙNG VỪA TẢI LÊN.
Chỉ trả lời bằng MỘT từ duy nhất: `rag_query` hoặc `history_query` hoặc `file_rag_query`.
---
[Tình trạng file]
{file_status}
---
[Lịch sử trò chuyện]
{chat_history}
---
[Câu hỏi mới]
{question}
---
Phân loại (chỉ 1 từ):
""")

HISTORY_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Bạn là trợ lý AI tại CUSC.
Chỉ dựa vào LỊCH SỬ TRÒ CHUYỆN được cung cấp, hãy trả lời CÂU HỎI của người dùng.
Không được bịa đặt thông tin.
---
Lịch sử trò chuyện:
{chat_history}
---
Câu hỏi: {question}
---
Câu trả lời:
""")

VISION_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Bạn là trợ lý AI. Nhiệm vụ của bạn là trả lời CÂU HỎI của người dùng.
Để trả lời, bạn phải sử dụng TẤT CẢ các thông tin sau:
1. HÌNH ẢNH được cung cấp.
2. LỊCH SỬ TRÒ CHUYỆN (để hiểu bối cảnh).
Hệ thống sẽ tự động tìm kiếm tài liệu (RAG) nếu cần.
Hãy phân tích HÌNH ẢNH, kết hợp thông tin tìm được (nếu có) và trả lời CÂU HỎI.
---
[Lịch sử trò chuyện]
{chat_history}
---
[Câu hỏi]
{question}
---
Câu trả lời chi tiết:
""")

RAG_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Bạn là trợ lý AI tại CUSC. Sử dụng công cụ tìm kiếm file được cung cấp để lấy thông tin liên quan từ tài liệu và trả lời câu hỏi của người dùng.

Lịch sử trò chuyện trước:
{chat_history}

Câu hỏi hiện tại: {question}

Trả lời câu hỏi dựa trên thông tin được lấy và lịch sử trò chuyện:
""")

FALLBACK_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Dựa trên lịch sử trò chuyện và câu hỏi, hãy cung cấp câu trả lời hữu ích.

Lịch sử:
{chat_history}

Câu hỏi: {question}

Trả lời:
""")


# --- HÀM VIẾT LẠI (create_rag_router_chain) VỚI LOGIC MỚI ---
def create_rag_router_chain(llm):
    """Tạo chain RAG có bộ định tuyến, sử dụng Google File Search Tool (SDK thô)."""
    if llm is None:
        print("Lỗi: Không thể tạo RAG chain do thiếu LLM.")
        return None

    def get_history_for_request(session_id: str, user_id: str):
        return get_session_history(session_id, user_id)

    # --- Router chain với format history ---
    router_chain = (
        {
            "file_status": lambda x: x["file_status"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
            "question": lambda x: x["question"]
        }
        | ROUTER_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    # --- History chain với format ---
    history_chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
        }
        | HISTORY_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    # --- Fallback chain với format ---
    base_llm_chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
        }
        | FALLBACK_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    # --- Logic Route (ĐÃ SỬA ĐỂ DÙN SDK THÔ VÀ INVOKE NGAY) ---
    def route(input_dict, config=None):
        session_id = config["configurable"]["session_id"]

        # 1. Kiểm tra tình trạng file
        user_file_store_name = get_session_file_store(session_id)
        file_status = "Người dùng đã tải lên 1 file." if user_file_store_name else "Người dùng CHƯA tải lên file nào."

        # 2. Chạy router
        try:
            classification = router_chain.invoke({
                "chat_history": input_dict.get("chat_history", []),
                "question": input_dict["question"],
                "file_status": file_status
            }, config)
        except Exception as e:
            classification = "rag_query"

        # 3. Trả về chain tương ứng (INVOKE NGAY)
        if "history_query" in classification:
            print("--- (Router: Lịch sử) ---")
            return history_chain.invoke(input_dict)

        # 4. Xác định store_name để sử dụng (ƯU TIÊN SESSION NẾU CÓ)
        store_to_use = None
        if user_file_store_name:
            print(f"--- (Router: File Search - Session Store: {user_file_store_name}) ---")
            store_to_use = user_file_store_name
        elif app_config.CUSC_MAIN_STORE_NAME:
            print(f"--- (Router: File Search - Main Store: {app_config.CUSC_MAIN_STORE_NAME}) ---")
            store_to_use = app_config.CUSC_MAIN_STORE_NAME

        if not store_to_use:
            print("--- (Router: Không có File Store - Trả lời bình thường) ---")
            return base_llm_chain.invoke(input_dict)

        # Raw SDK cho RAG với citations (INVOKE NGAY)
        def rag_raw_func(inputs):
            question = inputs["question"]
            chat_history = inputs["chat_history"]
            history_str = format_chat_history(chat_history)
            prompt_text = RAG_PROMPT_TEMPLATE.invoke({
                "chat_history": history_str,
                "question": question
            }).to_string()

            try:
                response = GLOBAL_GENAI_CLIENT.models.generate_content(
                    model=app_config.TEXT_MODEL_NAME,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[store_to_use]
                                )
                            )
                        ]
                    ),
                )
                text_response = response.text
                citations = extract_citations(response)
                return text_response + citations
            except Exception as e:
                return f"Lỗi khi tạo nội dung: {str(e)}"

        return rag_raw_func(input_dict)

    # --- Chain cơ sở có router (Giữ nguyên) ---
    base = (
        {"question": lambda x: x["question"],
         "chat_history": lambda x: x.get("chat_history", [])}
        | RunnableLambda(route)
    )

    # --- Bọc bộ nhớ (Giữ nguyên) ---
    chain_with_history = RunnableWithMessageHistory(
        base,
        get_history_for_request,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID"),
            ConfigurableFieldSpec(id="session_id", annotation=str, name="Session ID"),
        ]
    )
    return chain_with_history


# --- HÀM VIẾT LẠI (create_vision_chain) VỚI LOGIC MỚI ---
def create_vision_chain(llm):
    """Tạo chain Vision RAG, sử dụng Google File Search Tool (SDK thô)."""
    if llm is None:
        print("Lỗi: Không thể tạo Vision chain do thiếu LLM.")
        return None

    # --- Logic Route (MỚI - INVOKE NGAY) ---
    def route_vision(input_dict, config=None):
        session_id = config["configurable"]["session_id"]

        history = input_dict.get("chat_history", [])
        human_message_input = input_dict["question"]

        # 1. CHỌN TOOL RAG (ƯU TIÊN SESSION NẾU CÓ)
        store_to_use = None
        user_file_store_name = get_session_file_store(session_id)

        if user_file_store_name:
            print(f"--- (Vision: Gắn Session File Store {user_file_store_name}) ---")
            store_to_use = user_file_store_name
        elif app_config.CUSC_MAIN_STORE_NAME:
            print(f"--- (Vision: Gắn Main File Store {app_config.CUSC_MAIN_STORE_NAME}) ---")
            store_to_use = app_config.CUSC_MAIN_STORE_NAME
        else:
            print("--- (Vision: Không có File Store) ---")

        # 2. Extract question_text và image_base64 (SAFE CHECK)
        question_text = ""
        image_base64 = None
        image_parts = []
        if hasattr(human_message_input, 'content'):
            content = human_message_input.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            question_text = part.get("text", "")
                        elif part.get("type") == "image_url":
                            url = part["image_url"].get("url", "")
                            if url.startswith("data:image/jpeg;base64,"):
                                image_base64 = url.split(",")[1]
                            image_parts.append(part)  # Giữ nguyên cho langchain
            else:
                question_text = content
        else:
            # Fallback nếu là str
            question_text = str(human_message_input)
            image_base64 = None

        if not question_text:
            return "Lỗi: Không có câu hỏi."

        history_str = format_chat_history(history)
        prompt_text = VISION_PROMPT_TEMPLATE.invoke({
            "question": question_text,
            "chat_history": history_str,
        }).to_string()

        if store_to_use:
            # Raw SDK với tool và citations (INVOKE NGAY)
            def raw_vision_func(inputs):
                # Re-extract vì inputs giống input_dict (SAFE CHECK)
                hm_input = inputs["question"]
                img_b64 = None
                q_text = ""
                if hasattr(hm_input, 'content'):
                    content = hm_input.content
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    q_text = part.get("text", "")
                                elif part.get("type") == "image_url":
                                    url = part["image_url"].get("url", "")
                                    if url.startswith("data:image/jpeg;base64,"):
                                        img_b64 = url.split(",")[1]
                                    break
                    else:
                        q_text = content
                else:
                    # Fallback nếu str
                    q_text = str(hm_input)
                    img_b64 = None

                if not img_b64:
                    return "Lỗi: Không tìm thấy ảnh."

                hist_str = format_chat_history(inputs["chat_history"])

                p_text = VISION_PROMPT_TEMPLATE.invoke({
                    "question": q_text,
                    "chat_history": hist_str,
                }).to_string()

                contents = [
                    types.Part(text=p_text),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img_b64)
                        )
                    )
                ]

                try:
                    tool_config = types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[store_to_use]
                                )
                            )
                        ]
                    )
                    response = GLOBAL_GENAI_CLIENT.models.generate_content(
                        model=app_config.VISION_MODEL_NAME,
                        contents=contents,
                        config=tool_config
                    )
                    text_response = response.text
                    citations = extract_citations(response)
                    return text_response + citations
                except Exception as e:
                    return f"Lỗi khi tạo nội dung: {str(e)}"

            return raw_vision_func(input_dict)
        else:
            # LangChain không tool (INVOKE NGAY)
            def langchain_vision_func(inputs):
                hm_input = inputs["question"]
                # Extract (SAFE CHECK)
                q_text = ""
                img_parts = []
                if hasattr(hm_input, 'content'):
                    content = hm_input.content
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    q_text = part.get("text", "")
                                elif part.get("type") == "image_url":
                                    img_parts.append(part)
                    else:
                        q_text = content
                else:
                    q_text = str(hm_input)

                hist_str = format_chat_history(inputs["chat_history"])
                p_text = VISION_PROMPT_TEMPLATE.invoke({
                    "question": q_text,
                    "chat_history": hist_str,
                }).to_string()

                final_content = [{"type": "text", "text": p_text}] + img_parts
                final_hm = HumanMessage(content=final_content)

                try:
                    response = VISION_LLM.invoke(final_hm)
                    return response.content
                except Exception as e:
                    return f"Lỗi khi tạo nội dung: {str(e)}"

            return langchain_vision_func(input_dict)

    # --- Các hàm helper cho bộ nhớ (Giữ nguyên) ---
    def _format_history_input(input_dict):
        question = input_dict["question"]
        img_path = input_dict["image_path"]
        image_base64 = image_to_base64(img_path)
        if not image_base64: return HumanMessage(content=f"(Lỗi ảnh) {question}")
        image_data = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        return HumanMessage(content=[{"type": "text", "text": question}, image_data])

    def get_history_for_request(session_id: str, user_id: str):
        return get_session_history(session_id, user_id)

    # --- Chain cơ sở (MỚI) ---
    base_vision = RunnableLambda(route_vision)

    # --- Bọc bộ nhớ (Giữ nguyên) ---
    vision_chain_with_history = RunnableWithMessageHistory(
        base_vision,
        get_history_for_request,
        input_messages_key="question",
        input_messages_key_fx=RunnableLambda(_format_history_input),
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID"),
            ConfigurableFieldSpec(id="session_id", annotation=str, name="Session ID"),
        ]
    )
    return vision_chain_with_history


# ==============================================================================
# SECTION 4: KHỞI TẠO CHAIN TOÀN CỤC (ĐỂ API SỬ DỤNG)
# ==============================================================================

RAG_CHAIN_WITH_HISTORY = create_rag_router_chain(TEXT_LLM)
VISION_CHAIN_WITH_HISTORY = create_vision_chain(VISION_LLM)


# ==============================================================================
# SECTION 5: CÁC HÀM XỬ LÝ CLI (COMMAND-LINE INTERFACE)
# ==============================================================================

# (Các hàm này giữ nguyên, chúng không cần thay đổi)
def handle_text_query(query_text, user_id, session_id="default_session"):
    print("--- 🔍 Đang xử lý câu hỏi văn bản bằng RAG ---")
    chain_to_run = RAG_CHAIN_WITH_HISTORY
    if chain_to_run is None:
        return
    full_response = ""
    config_ = {"configurable": {"session_id": session_id, "user_id": user_id}}
    input_data = {"question": query_text}
    try:
        # Sử dụng invoke thay vì stream vì route trả về str trực tiếp
        response = chain_to_run.invoke(input_data, config=config_)
        full_response = str(response)
        print(full_response)
        print("\n")
        save_session_message(session_id, user_id, query_text, full_response)
    except Exception as e:
        print(f"\nLỗi khi xử lý câu hỏi text: {e}")


def handle_multimodal_query(query_text, image_path, user_id, session_id="default_session"):
    print(f"--- 🖼️ Xử lý câu hỏi có ảnh: {os.path.basename(image_path)} ---")
    chain_to_run = VISION_CHAIN_WITH_HISTORY
    if chain_to_run is None:
        return
    full_response = ""
    input_data = {"question": query_text, "image_path": image_path}
    config_ = {"configurable": {"session_id": session_id, "user_id": user_id}}
    try:
        # Sử dụng invoke thay vì stream vì route trả về str trực tiếp
        response = chain_to_run.invoke(input_data, config=config_)
        full_response = str(response)
        print(full_response)
        print("\n")
        save_session_message(session_id, user_id, query_text, full_response, image_path=image_path)
    except Exception as e:
        print(f"\nLỗi khi xử lý câu hỏi ảnh: {e}")


def handle_pdf_upload(pdf_path: str, session_id: str, user_id: str):
    print(f"\n⏳ Đang xử lý file: {pdf_path}...")
    try:
        file_id = save_pdf_to_mongo(pdf_path, session_id, user_id)
        if file_id:
            process_and_vectorize_pdf(pdf_path, session_id, user_id)  # Hàm đã refactor
            print("✅ Xử lý và tải file lên Google thành công.")
        else:
            print("❌ Lỗi khi lưu file vào DB.")
    except Exception as ex:
        print(f"❌ Lỗi nghiêm trọng khi xử lý file PDF: {ex}")


# ==============================================================================
# SECTION 6: HÀM MAIN CHO CLI (Giữ nguyên)
# ==============================================================================

def main():
    print("🤖 Chatbot CUSC (Google File Search) sẵn sàng!")
    print("=" * 30)
    print("[1] Tạo session mới")
    print("[2] Tiếp tục session cũ")

    user_id = "6910c339c0f7d8f23ecc1cc4"  # User ID ví dụ
    choice = input("Lựa chọn của bạn (1 hoặc 2): ").strip()
    if choice == '2':
        print("\nĐang tải các session gần đây...")
        sessions = list_sessions(limit=10, user_id=user_id)
        if not sessions:
            print("Không tìm thấy session nào. Sẽ tạo session mới.")
            session_id = str(uuid.uuid4())
        else:
            for i, s in enumerate(sessions):
                print(f"  [{i + 1}] {s['session_id']} ({s['num_messages']} tin nhắn, cập nhật: {s['updated_at']})")
            try:
                s_choice = int(input("Chọn session (nhập số 1, 2,...) hoặc 0 để tạo mới: ").strip())
                if 0 < s_choice <= len(sessions):
                    session_id = sessions[s_choice - 1]['session_id']
                else:
                    session_id = str(uuid.uuid4())
            except ValueError:
                session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())
    print(f"\n🆔 Session ID hiện tại: {session_id}")
    print("   Gõ 'exit' để thoát.")
    print("   Gõ 'pdf' để tải file PDF mới.\n")
    get_session_history(session_id, user_id)
    while True:
        print("-" * 20)
        user_input = input("👤 Bạn hỏi (hoặc gõ 'pdf'): ")
        if user_input.lower() == "exit":
            print("Tạm biệt!")
            break
        if user_input.lower() == "pdf":
            pdf_path = input("📂 Nhập đường dẫn PDF: ").strip()
            if pdf_path and os.path.exists(pdf_path):
                handle_pdf_upload(pdf_path, session_id, user_id)
            else:
                print(f"⚠️ Không tìm thấy file tại '{pdf_path}'")
            continue
        query_text = user_input
        image_path = input("🖼️ Nhập đường dẫn ảnh (Enter nếu không có): ").strip()
        print("\n💡 Trả lời:")
        if image_path and os.path.exists(image_path):
            handle_multimodal_query(query_text, image_path, user_id, session_id)
        elif image_path:
            print(f"⚠️ Không tìm thấy ảnh tại '{image_path}'")
        else:
            handle_text_query(query_text, user_id, session_id)


if __name__ == "__main__":
    main()  # Mai vào Grok copy rồi sửa logic (vision + file pdf)
