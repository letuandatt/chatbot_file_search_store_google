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


def extract_citations(response, show_details=False):
    """Trích xuất và format nguồn trích dẫn từ response metadata.

    Args:
        response: Response từ Google AI
        show_details: Nếu True, hiển thị thêm thông tin chi tiết (số đoạn trích dẫn)
    """
    try:
        metadata = response.candidates[0].grounding_metadata
        if not (metadata and metadata.grounding_supports and metadata.grounding_chunks):
            return ""  # Không có trích dẫn

        all_chunks = metadata.grounding_chunks
        file_citation_count = {}

        # Đếm số lượng trích dẫn cho mỗi file
        for support in metadata.grounding_supports:
            for chunk_index in support.grounding_chunk_indices:
                if 0 <= chunk_index < len(all_chunks):
                    chunk = all_chunks[chunk_index]
                    filename = chunk.retrieved_context.title
                    file_citation_count[filename] = file_citation_count.get(filename, 0) + 1

        if not file_citation_count:
            return ""

        # Format phần citations
        citations_str = "\n\n--- 📚 Nguồn tham khảo ---\n"
        for filename, count in file_citation_count.items():
            if show_details:
                citations_str += f"📄 {filename} (trích dẫn {count} đoạn)"
            else:
                citations_str += f"📄 {filename}"

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
                        "user_id": user_id,
                        "upload_time": now
                    }
                )
        except Exception as img_ex:
            print(f"Lỗi khi lưu ảnh vào GridFS: {img_ex}")

    message_data = {
        "question": question,
        "answer": answer,
        "image_gridfs_id": str(image_gridfs_id) if image_gridfs_id else None,
        "timestamp": now
    }

    coll.update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": message_data},
            "$set": {"updated_at": now},
            "$setOnInsert": {
                "user_id": user_id,
                "created_at": now
            }
        },
        upsert=True
    )


def load_session_messages(session_id, user_id, limit=100):
    """Load messages của session cụ thể theo user_id."""
    coll = get_mongo_collection()
    if coll is None:
        return InMemoryChatMessageHistory()
    session = coll.find_one({"session_id": session_id, "user_id": user_id})
    if session and "messages" in session:
        memory = InMemoryChatMessageHistory()
        for msg in session["messages"][-limit:]:
            question = msg.get("question", "")
            answer = msg.get("answer", "")
            if question:
                memory.add_message(HumanMessage(content=question))
            if answer:
                memory.add_message(AIMessage(content=answer))
        return memory
    return InMemoryChatMessageHistory()


def list_sessions(limit=20, user_id=None):
    """Lấy danh sách session từ MongoDB, lọc theo user_id nếu có."""
    coll = get_mongo_collection()
    if coll is None:
        print("Lỗi: MongoDB chưa kết nối.")
        return []
    query = {"user_id": user_id} if user_id else {}
    sessions = coll.find(query, projection={"session_id": 1, "created_at": 1, "updated_at": 1, "user_id": 1,
                                            "messages": 1}).sort("updated_at", DESCENDING).limit(limit)
    result_list = []
    for s in sessions:
        num_msgs = len(s.get("messages", []))
        result_list.append({"session_id": s["session_id"], "created_at": s.get("created_at", "N/A"),
                            "updated_at": s.get("updated_at", "N/A"), "user_id": s.get("user_id", "N/A"),
                            "num_messages": num_msgs})
    return result_list


def list_documents_by_user(user_id: str, limit: int = 50):
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        print("Lỗi: MongoDB chưa kết nối.")
        return []
    try:
        docs_cursor = coll.find(
            {"user_id": user_id},
            projection={
                "_id": 1,
                "session_id": 1,
                "filename": 1,
                "created_at": 1,
                "status": 1,
                "file_store_name": 1,
                "file_hash": 1
            }
        ).sort("created_at", DESCENDING).limit(limit)
        documents = []
        for doc in docs_cursor:
            documents.append({
                "id": str(doc["_id"]),
                "session_id": doc.get("session_id", "N/A"),
                "filename": doc.get("filename", "N/A"),
                "created_at": doc.get("created_at", "N/A"),
                "status": doc.get("status", "N/A"),
                "file_store_name": doc.get("file_store_name", ""),
                "file_hash": doc.get("file_hash", "")
            })
        return documents
    except Exception as e:
        print(f"Lỗi khi lấy danh sách documents theo user: {e}")
        return []


def get_session_file_store(session_id: str) -> str | None:
    """LẤY FILE STORE CỦA SESSION - FIXED: Kiểm tra None trước khi dùng"""
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return None
    try:
        doc_record = coll.find_one(
            {"session_id": session_id, "status": "processed"},
            projection={"file_store_name": 1}
        )
        if doc_record and "file_store_name" in doc_record:
            return doc_record.get("file_store_name")
        return None
    except Exception as e:
        print(f"Lỗi khi lấy session file store: {e}")
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
    try:
        file_hash = compute_file_hash(file_path)
        # Kiểm tra file đã tồn tại và xử lý xong chưa
        existing = coll.find_one({"file_hash": file_hash, "user_id": user_id, "status": "processed"})
        if existing:
            print(f"File đã được tải lên và xử lý. File Store: {existing.get('file_store_name')}")
            coll.update_one(
                {"_id": existing["_id"]},
                {"$addToSet": {"sessions": session_id}}
            )
            return str(existing["_id"])
        now = datetime.now(VN_TZ).isoformat()
        with open(file_path, "rb") as f:
            file_id = fs_client.put(f, filename=os.path.basename(file_path))
        doc_data = {
            "user_id": user_id,
            "session_id": session_id,
            "sessions": [session_id],
            "filename": os.path.basename(file_path),
            "file_gridfs_id": str(file_id),
            "file_hash": file_hash,
            "created_at": now,
            "status": "uploaded"
        }
        result = coll.insert_one(doc_data)
        print(f"Đã lưu file vào DB với document ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"Lỗi khi lưu file vào DB: {e}")
        return None


def process_and_vectorize_pdf(file_path: str, session_id: str, user_id: str):
    """
    Upload PDF lên Google File Search Tool, tạo File Store tự động cho session.
    """
    coll = DB_DOCUMENTS_COLLECTION
    client = GLOBAL_GENAI_CLIENT
    if coll is None or client is None:
        print("Lỗi: Thiếu MongoDB hoặc Google AI client.")
        return

    file_name = os.path.basename(file_path)
    print(f"Đang xử lý file {file_name} với Google File Search Tool...")

    try:
        store_display_name = f"session-store-{session_id[:16]}-{uuid.uuid4().hex[:12]}"
        file_store = client.file_search_stores.create(
            config={'display_name': store_display_name}
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
            session_delete = sessions_coll.delete_one({"_id": session_doc["_id"]})
            deleted_counts["sessions"] = session_delete.deleted_count
            print(f"Đã xóa session '{session_id}' khỏi collection 'sessions'.")
    except Exception as e:
        print(f"Lỗi khi xóa session: {e}")

    try:
        doc_records = docs_coll.find({"session_id": session_id, "user_id": user_id})
        for doc in doc_records:
            if doc.get("file_gridfs_id"):
                try:
                    gridfs_ids_to_delete.append(ObjectId(doc["file_gridfs_id"]))
                except Exception:
                    pass
            if doc.get("file_store_name"):
                file_store_names_to_delete.add(doc["file_store_name"])
        doc_delete = docs_coll.delete_many({"session_id": session_id, "user_id": user_id})
        deleted_counts["document_records"] = doc_delete.deleted_count
        print(f"Đã xóa {deleted_counts['document_records']} document records của session.")
    except Exception as e:
        print(f"Lỗi khi xóa document records: {e}")

    for gf_id in gridfs_ids_to_delete:
        try:
            fs_client.delete(gf_id)
            deleted_counts["gridfs_files"] += 1
        except Exception as e:
            print(f"Lỗi khi xóa GridFS file {gf_id}: {e}")

    for store_name in file_store_names_to_delete:
        try:
            client.file_search_stores.delete(name=store_name)
            deleted_counts["file_stores"] += 1
            print(f"Đã xóa File Store: {store_name}")
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

    # --- Chain nhánh cơ sở (History Path, Fallback) ---
    history_chain = HISTORY_PROMPT_TEMPLATE | llm | StrOutputParser()
    base_llm_chain = FALLBACK_PROMPT_TEMPLATE | llm | StrOutputParser()

    # --- Logic Route (MỚI) ---
    def route(input_dict, config=None):
        session_id = config["configurable"]["session_id"]
        user_id = config["configurable"]["user_id"]
        question = input_dict["question"]
        chat_history = input_dict["chat_history"]

        # FIXED: Kiểm tra user ownership của session
        if not check_session_belongs_to_user(session_id, user_id):
            return "Lỗi: Session không thuộc về user này."

        # --- 1. DETECT INTENT ---
        file_status = "Không có tài liệu cụ thể"
        user_file_store_name = get_session_file_store(session_id)
        if user_file_store_name:
            file_status = f"Người dùng đã tải lên tài liệu cho session này (Store: {user_file_store_name})"

        file_keywords = ["file", "tài liệu", "tập tin", "pdf", "vừa tải", "đã tải", "upload", "đọc file"]
        is_file_question = any(kw.lower() in question.lower() for kw in file_keywords)

        # Route qua file store nếu có từ khóa file VÀ có file store
        if is_file_question and user_file_store_name:
            print(f"--- (Router: File Search - Session Store: {user_file_store_name}) ---")
            store_to_use = user_file_store_name
        elif not is_file_question and app_config.CUSC_MAIN_STORE_NAME:
            # Câu hỏi chung - dùng main store
            print(f"--- (Router: General RAG - Main Store: {app_config.CUSC_MAIN_STORE_NAME}) ---")
            store_to_use = app_config.CUSC_MAIN_STORE_NAME
        elif is_file_question and not user_file_store_name:
            # User hỏi về file nhưng chưa upload
            print("--- (Router: User hỏi về file nhưng chưa upload) ---")
            return "Bạn chưa tải lên tài liệu nào cho session này. Vui lòng tải file PDF trước khi hỏi."
        else:
            # Không có store nào - trả lời bình thường
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
                # FIXED: Kiểm tra GLOBAL_GENAI_CLIENT không phải None
                if GLOBAL_GENAI_CLIENT is None:
                    return "Lỗi: Google AI client chưa được khởi tạo."

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
                # FIXED: Kiểm tra response và response.text tồn tại
                if response and hasattr(response, 'text'):
                    text_response = response.text if response.text else "Không thể tạo câu trả lời."
                else:
                    text_response = "Không thể tạo câu trả lời."

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
        user_id = config["configurable"]["user_id"]

        # FIXED: Kiểm tra user ownership của session
        if not check_session_belongs_to_user(session_id, user_id):
            return "Lỗi: Session không thuộc về user này."

        history = input_dict.get("chat_history", [])
        # FIXED: Xử lý cả HumanMessage và image_path
        if "image_path" in input_dict:
            # Input từ CLI với image_path
            image_path = input_dict["image_path"]
            question_text = input_dict["question"]

            # Kiểm tra file ảnh tồn tại
            if not os.path.exists(image_path):
                return f"Lỗi: Không tìm thấy ảnh tại '{image_path}'"

            image_base64 = image_to_base64(image_path)
            if not image_base64:
                return "Lỗi: Không thể xử lý ảnh."
        else:
            # Input từ API/chain với HumanMessage
            human_message_input = input_dict["question"]

            # Extract question_text và image_base64 từ HumanMessage
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
                                image_parts.append(part)
                else:
                    question_text = content
            else:
                question_text = str(human_message_input)

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

        if not question_text:
            return "Lỗi: Không có câu hỏi."

        if not image_base64:
            return "Lỗi: Không tìm thấy ảnh."

        history_str = format_chat_history(history)
        prompt_text = VISION_PROMPT_TEMPLATE.invoke({
            "question": question_text,
            "chat_history": history_str,
        }).to_string()

        if store_to_use:
            # Raw SDK với tool và citations
            try:
                # FIXED: Kiểm tra GLOBAL_GENAI_CLIENT
                if GLOBAL_GENAI_CLIENT is None:
                    return "Lỗi: Google AI client chưa được khởi tạo."

                contents = [
                    types.Part(text=prompt_text),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(image_base64)
                        )
                    )
                ]

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
                # FIXED: Kiểm tra response
                if response and hasattr(response, 'text'):
                    text_response = response.text if response.text else "Không thể tạo câu trả lời."
                else:
                    text_response = "Không thể tạo câu trả lời."

                citations = extract_citations(response)
                return text_response + citations
            except Exception as e:
                return f"Lỗi khi tạo nội dung: {str(e)}"
        else:
            # LangChain không tool
            try:
                # Tạo HumanMessage với cả text và image
                final_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                final_hm = HumanMessage(content=final_content)
                response = VISION_LLM.invoke(final_hm)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"Lỗi khi tạo nội dung: {str(e)}"

    # --- Các hàm helper cho bộ nhớ ---
    def _format_history_input(input_dict):
        # FIXED: Xử lý cả trường hợp có image_path
        if "image_path" in input_dict:
            question = input_dict["question"]
            img_path = input_dict["image_path"]
            if not os.path.exists(img_path):
                return HumanMessage(content=f"(Lỗi ảnh) {question}")
            image_base64 = image_to_base64(img_path)
            if not image_base64:
                return HumanMessage(content=f"(Lỗi ảnh) {question}")
            image_data = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            return HumanMessage(content=[{"type": "text", "text": question}, image_data])
        else:
            # Trường hợp đã có HumanMessage
            return input_dict["question"]

    def get_history_for_request(session_id: str, user_id: str):
        return get_session_history(session_id, user_id)

    # --- Chain cơ sở ---
    base_vision = RunnableLambda(route_vision)

    # --- Bọc bộ nhớ ---
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
    main()