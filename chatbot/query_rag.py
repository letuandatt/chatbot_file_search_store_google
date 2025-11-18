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
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# --- Agent Imports ---
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

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

# --- GOOGLE AI SDK CLIENT ---
try:
    GLOBAL_GENAI_CLIENT = genai.Client(api_key=app_config.GOOGLE_API_KEY)
    print("Google Generative AI SDK client initialized.")
except Exception as e:
    print(f"Lỗi khi cấu hình Google AI SDK client: {e}")
    GLOBAL_GENAI_CLIENT = None


def get_mongo_collection(collection_name: str = "sessions"):
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
        # 1. LẤY DANH SÁCH INDEX HIỆN TẠI
        existing_indexes = DB_DOCUMENTS_COLLECTION.index_information()

        # 2. XÓA INDEX CŨ NẾU LÀ UNIQUE TRÊN file_hash
        for idx_name, idx_info in existing_indexes.items():
            if idx_name.startswith("file_hash_") and idx_info.get("unique", False):
                print(f"Đang xóa index unique cũ: {idx_name}")
                DB_DOCUMENTS_COLLECTION.drop_index(idx_name)

        # 3. TẠO LẠI INDEX THƯỜNG (không unique) ĐỂ QUERY NHANH
        DB_DOCUMENTS_COLLECTION.create_index([("file_hash", 1)], name="file_hash_idx", background=True)

        # 4. Các index khác
        desired_indexes = [
            ("session_id", ASCENDING),
            ("user_id", ASCENDING),
            ("created_at", DESCENDING),
        ]
        for key_tuple in desired_indexes:
            field_name, direction = (key_tuple, ASCENDING) if not isinstance(key_tuple, tuple) else key_tuple
            index_name = f"{field_name}_{direction}"
            try:
                DB_DOCUMENTS_COLLECTION.create_index([(field_name, direction)], name=index_name, background=True)
            except Exception:
                pass

        print("MongoDB collection 'documents' indexes đã được sửa hoàn toàn (file_hash không còn unique).")
except Exception as e:
    print(f"Failed to initialize 'documents' collection: {e}")
    DB_DOCUMENTS_COLLECTION = None


def check_session_belongs_to_user(session_id: str, user_id: str) -> bool:
    coll = get_mongo_collection("sessions")
    if coll is None:
        return False
    try:
        return coll.count_documents({"session_id": session_id, "user_id": user_id}, limit=1) > 0
    except Exception:
        return False


# --- VIETNAM TIMEZONE ---
try:
    VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
except pytz.UnknownTimeZoneError:
    VN_TZ = timezone.utc


# --- LLM MODEL ---
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
# SECTION 2: CÁC HÀM TIỆN ÍCH CỐT LÕI
# ==============================================================================
def format_chat_history(history):
    if not history: return ""
    formatted_parts = []
    for message in history:
        role = getattr(message, 'role', str(type(message).__name__))
        content = getattr(message, 'content', str(message))
        if isinstance(content, list):
            text_content = ""
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content = part.get("text", "")
                    break
            content = text_content
        formatted_parts.append(f"{role.upper()}: {content}")
    return "\n".join(formatted_parts)


def extract_citations(response, show_details=False):
    try:
        metadata = response.candidates[0].grounding_metadata
        if not (metadata and metadata.grounding_supports and metadata.grounding_chunks):
            return ""
        all_chunks = metadata.grounding_chunks
        file_citation_count = {}
        for support in metadata.grounding_supports:
            for chunk_index in support.grounding_chunk_indices:
                if 0 <= chunk_index < len(all_chunks):
                    chunk = all_chunks[chunk_index]
                    filename = chunk.retrieved_context.title
                    file_citation_count[filename] = file_citation_count.get(filename, 0) + 1
        if not file_citation_count: return ""
        citations_str = "\n\n--- 📚 Nguồn tham khảo ---\n"
        for filename, count in file_citation_count.items():
            citations_str += f"📄 {filename}" + (f" ({count} đoạn)" if show_details else "") + "\n"
        return citations_str
    except Exception:
        return ""


def save_session_message(session_id, user_id, question, answer, image_path=None):
    coll = get_mongo_collection()
    fs_client = FS

    if coll is None:
        return

    now = datetime.now(VN_TZ).isoformat()
    image_gridfs_id = None
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as i_f:
                image_gridfs_id = fs_client.put(i_f, filename=os.path.basename(image_path),
                                                metadata={
                                                    "session_id": session_id,
                                                    "user_id": user_id
                                                })
        except Exception:
            pass

    message_data = {
        "question": question,
        "answer": answer,
        "image_gridfs_id": str(image_gridfs_id) if image_gridfs_id else None, "timestamp": now
    }

    coll.update_one({
        "session_id": session_id},
        {
            "$push": {
                "messages": message_data
            },
            "$set": {
                "updated_at": now
            },
            "$setOnInsert": {
                "user_id": user_id,
                "created_at": now
            }
        }, upsert=True)


def load_session_messages(session_id, user_id, limit=100):
    coll = get_mongo_collection()
    if coll is None: return InMemoryChatMessageHistory()
    session = coll.find_one({
        "session_id": session_id,
        "user_id": user_id
    })
    memory = InMemoryChatMessageHistory()
    if session and "messages" in session:
        for msg in session["messages"][-limit:]:
            if msg.get("question"):
                memory.add_message(HumanMessage(content=msg.get("question")))
            if msg.get("answer"):
                memory.add_message(AIMessage(content=msg.get("answer")))
    return memory


def list_sessions(limit=20, user_id=None):
    coll = get_mongo_collection()
    if coll is None: return []
    query = {"user_id": user_id} if user_id else {}
    sessions = coll.find(
        query,
        projection={
            "session_id": 1,
            "created_at": 1,
            "updated_at": 1,
            "user_id": 1,
            "messages": 1
        }).sort("updated_at", DESCENDING).limit(limit)

    return [{
        "session_id": s["session_id"],
        "created_at": s.get("created_at", "N/A"),
        "updated_at": s.get("updated_at", "N/A"),
        "num_messages": len(s.get("messages", []))
    } for s in sessions]


def get_session_file_stores(session_id: str) -> list[str]:
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return []
    try:
        cursor = coll.find({"session_id": session_id, "status": "processed"}, {"file_store_name": 1})
        return [doc.get("file_store_name") for doc in cursor if doc.get("file_store_name")]
    except Exception:
        return []


def compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f: return hashlib.md5(f.read()).hexdigest()


def save_pdf_to_mongo(file_path: str, session_id: str, user_id: str) -> str | None:
    fs_client = FS
    coll = DB_DOCUMENTS_COLLECTION
    if fs_client is None or coll is None:
        return None
    try:
        file_hash = compute_file_hash(file_path)
        file_name = os.path.basename(file_path)

        existing_in_session = coll.find_one({"file_hash": file_hash, "user_id": user_id, "session_id": session_id})
        if existing_in_session:
            return str(existing_in_session["_id"])

        hash_existing = coll.find_one({"file_hash": file_hash, "user_id": user_id})
        if hash_existing:
            file_gridfs_id = hash_existing["file_gridfs_id"]
            print(f"File đã tồn tại trước đó (user-wide), tái sử dụng GridFS ID: {file_gridfs_id}")
        else:
            with open(file_path, "rb") as f:
                file_id = fs_client.put(f, filename=file_name, metadata={"original_user": user_id})
            file_gridfs_id = str(file_id)
            print("File mới → đã upload lên GridFS.")

        result = coll.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "filename": file_name,
            "file_gridfs_id": file_gridfs_id,
            "file_hash": file_hash,
            "created_at": datetime.now(VN_TZ).isoformat(),
            "status": "uploaded"
        })
        return str(result.inserted_id)
    except Exception as e:
        print(f"Lỗi lưu file: {e}")
        return None


def process_and_vectorize_pdf(file_path: str, session_id: str, doc_id: str):
    coll = DB_DOCUMENTS_COLLECTION
    client = GLOBAL_GENAI_CLIENT

    if coll is None or client is None:
        return

    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    try:
        store_display_name = f"session-{session_id[:8]}-file-{doc_id[:8]}-{uuid.uuid4().hex[:8]}"
        file_store = client.file_search_stores.create(
            config={
                'display_name': store_display_name,
            }
        )
        client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=file_store.name,
            config={
                'display_name': file_name
            }
        )
        coll.update_one({"_id": ObjectId(doc_id)},
                        {"$set": {"status": "processed", "file_store_name": file_store.name}})
        print(f"Processed {file_name} -> Store: {file_store.name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        coll.update_one({"_id": ObjectId(doc_id)}, {"$set": {"status": "error_processing", "error": str(e)}})


def delete_session_and_associated_files(session_id: str, user_id: str) -> dict:
    sessions_coll = get_mongo_collection("sessions")
    docs_coll = DB_DOCUMENTS_COLLECTION
    fs_client = FS
    client = GLOBAL_GENAI_CLIENT
    if not all([sessions_coll, fs_client, client, docs_coll]): raise Exception("DB not ready")
    deleted = {"sessions": 0, "document_records": 0, "gridfs_files": 0, "file_stores": 0}
    try:
        session_doc = sessions_coll.find_one({"session_id": session_id, "user_id": user_id})
        if session_doc:
            sessions_coll.delete_one({"_id": session_doc["_id"]})
            deleted["sessions"] = 1
    except Exception:
        pass

    gridfs_ids, store_names = [], set()
    try:
        docs = docs_coll.find({"session_id": session_id, "user_id": user_id})
        for d in docs:
            if d.get("file_gridfs_id"): gridfs_ids.append(ObjectId(d["file_gridfs_id"]))
            if d.get("file_store_name"): store_names.add(d["file_store_name"])
        res = docs_coll.delete_many({"session_id": session_id, "user_id": user_id})
        deleted["document_records"] = res.deleted_count
    except Exception:
        pass

    for gid in gridfs_ids:
        try:
            fs_client.delete(gid); deleted["gridfs_files"] += 1
        except Exception:
            pass
    for sname in store_names:
        try:
            client.file_search_stores.delete(name=sname); deleted["file_stores"] += 1
        except Exception:
            pass
    return deleted


def image_to_base64(image_path, max_size_px=1024, jpeg_quality=85):
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size_px, max_size_px))
            if img.mode != 'RGB': img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None


def get_history_for_langchain(session_id: str, user_id: str):
    return load_session_messages(session_id, user_id)


get_session_history = get_history_for_langchain


# ==============================================================================
# SECTION 3: AGENT TOOLS & SETUP (NEW ARCHITECTURE)
# ==============================================================================

@tool
def tool_search_general_policy(query: str):
    """
    Sử dụng công cụ này để tra cứu các thông tin về quy trình, quy định, thủ tục, hoặc thông tin chung của CUSC (Đại học Cần Thơ).
    Tuyệt đối không dùng công cụ này để tìm kiếm trong file người dùng tải lên.
    """
    if not app_config.CUSC_MAIN_STORE_NAME:
        return "Hệ thống chưa được cấu hình Main Store."

    print(f"--- [Agent Tool] Searching General Policy: {query} ---")
    try:
        response = GLOBAL_GENAI_CLIENT.models.generate_content(
            model=app_config.TEXT_MODEL_NAME,
            contents=[types.Part(text=query)],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[app_config.CUSC_MAIN_STORE_NAME]
                    )
                )]
            ),
        )
        text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
        citations = extract_citations(response)
        return (text or "Không tìm thấy thông tin.") + citations
    except Exception as e:
        return f"Lỗi khi tra cứu quy trình chung: {str(e)}"


@tool
def tool_search_uploaded_file(query: str, session_id: str):
    """
    Sử dụng công cụ này KHI VÀ CHỈ KHI người dùng hỏi về nội dung của 'file', 'tài liệu', 'pdf'
    mà họ vừa tải lên trong phiên làm việc hiện tại.
    BẮT BUỘC phải truyền tham số `session_id` (lấy từ context).
    """
    if not session_id:
        return "Lỗi: Không có session_id để tìm kiếm file."

    print(f"--- [Agent Tool] Searching Uploaded File (Session: {session_id}): {query} ---")

    # Lấy danh sách stores của session này
    user_file_stores = get_session_file_stores(session_id)

    if not user_file_stores:
        return "Người dùng chưa tải lên tài liệu nào trong phiên này."

    try:
        response = GLOBAL_GENAI_CLIENT.models.generate_content(
            model=app_config.TEXT_MODEL_NAME,
            contents=[types.Part(text=query)],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=user_file_stores
                    )
                )]
            ),
        )
        text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
        citations = extract_citations(response)
        return (text or "Không tìm thấy thông tin trong file.") + citations
    except Exception as e:
        return f"Lỗi khi tra cứu file tải lên: {str(e)}"


# --- AGENT SYSTEM PROMPT ---
AGENT_SYSTEM_PROMPT = """
Bạn là trợ lý AI thông minh của CUSC (Can Tho University Software Center).
Nhiệm vụ của bạn là hỗ trợ nhân viên giải đáp thắc mắc và xử lý thông tin.

Bạn có quyền truy cập vào các công cụ sau:
1. `tool_search_general_policy`: Tra cứu quy định, quy trình chung (General Knowledge).
2. `tool_search_uploaded_file`: Tra cứu nội dung trong file PDF người dùng vừa tải lên.

HƯỚNG DẪN XỬ LÝ:
- **Ưu tiên ngữ cảnh:** Luôn xem xét Lịch sử trò chuyện.
- **Không lạm dụng tool:** Nếu là chào hỏi xã giao (ví dụ: "xin chào"), hãy trả lời thân thiện mà KHÔNG cần tool.
- **Chọn tool đúng:**
    - Nếu hỏi về quy trình/quy định chung: Dùng `tool_search_general_policy`.
    - Nếu hỏi về file/tài liệu vừa gửi: Dùng `tool_search_uploaded_file`.
- **QUAN TRỌNG:** Khi gọi `tool_search_uploaded_file`, bạn **BẮT BUỘC** phải cung cấp tham số `session_id`. 
  (Giá trị `session_id` được cung cấp trong [System Note] ở cuối câu hỏi của người dùng).
- **Trả lời:** Dựa vào kết quả tool trả về (nếu có) để trả lời người dùng bằng tiếng Việt.
"""


def create_agent_executor(llm):
    """Tạo Agent Executor thay thế cho Router Chain."""
    if llm is None:
        print("Lỗi: Không thể tạo Agent do thiếu LLM.")
        return None

    # 1. Bind Tools
    tools = [tool_search_general_policy, tool_search_uploaded_file]

    # 2. Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input_with_session_id}"),  # Input đã được inject session_id
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 3. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 4. Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    # 5. Helper: Prepare Input (Inject session_id)
    def _prepare_agent_input(input_dict, config):
        session_id = config.get("configurable", {}).get("session_id")
        # Tiêm session_id vào câu hỏi để Agent nhìn thấy
        input_with_session_id = (
            f"{input_dict['question']}\n\n"
            f"[System Note: current session_id is: {session_id}]"
        )
        return {
            "input_with_session_id": input_with_session_id,
            "chat_history": input_dict.get("chat_history", [])
        }

    # 6. Helper: Get History
    def get_history_wrapper(session_id: str, user_id: str):
        return get_session_history(session_id, user_id)

    # 7. Wrap with Input Prep & History
    agent_chain = (
            RunnablePassthrough()
            | RunnableLambda(_prepare_agent_input)
            | agent_executor
    )

    agent_with_history = RunnableWithMessageHistory(
        agent_chain,
        get_history_wrapper,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID"),
            ConfigurableFieldSpec(id="session_id", annotation=str, name="Session ID"),
        ]
    )

    return agent_with_history


# --- VISION CHAIN (Giữ nguyên logic Multimodal) ---
VISION_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Bạn là trợ lý AI. Trả lời câu hỏi dựa trên HÌNH ẢNH và LỊCH SỬ TRÒ CHUYỆN.
Nếu cần thiết, hệ thống sẽ cung cấp thêm thông tin từ tài liệu.
---
[Lịch sử trò chuyện]
{chat_history}
---
[Câu hỏi]
{question}
---
Câu trả lời chi tiết:
""")


def create_vision_chain(llm):
    if llm is None: return None

    def route_vision(input_dict, config=None):
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        user_id = configurable.get("user_id")
        if not check_session_belongs_to_user(session_id, user_id):
            return "Lỗi: Session không hợp lệ."

        history = input_dict.get("chat_history", [])

        # Extract Image & Question
        if "image_path" in input_dict:
            image_path, question_text = input_dict["image_path"], input_dict["question"]
            if not os.path.exists(image_path): return "Lỗi: Không tìm thấy file ảnh."
            image_base64 = image_to_base64(image_path)
        else:
            # Handle HumanMessage input (nếu dùng API)
            hm = input_dict["question"]
            question_text, image_base64 = "", None
            if hasattr(hm, 'content') and isinstance(hm.content, list):
                for p in hm.content:
                    if p.get("type") == "text":
                        question_text = p.get("text", "")
                    elif p.get("type") == "image_url":
                        image_base64 = p["image_url"]["url"].split(",")[1]

        if not image_base64: return "Lỗi: Không tìm thấy dữ liệu ảnh."

        # Chọn Store (Ưu tiên Session -> Main)
        user_stores = get_session_file_stores(session_id)
        stores_to_use = user_stores if user_stores else (
            [app_config.CUSC_MAIN_STORE_NAME] if app_config.CUSC_MAIN_STORE_NAME else None)

        prompt_text = VISION_PROMPT_TEMPLATE.invoke(
            {"question": question_text, "chat_history": format_chat_history(history)}).to_string()

        try:
            if GLOBAL_GENAI_CLIENT and stores_to_use:
                # Dùng Raw SDK nếu có file store (để RAG)
                response = GLOBAL_GENAI_CLIENT.models.generate_content(
                    model=app_config.VISION_MODEL_NAME,
                    contents=[
                        types.Part(text=prompt_text),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=base64.b64decode(image_base64)))
                    ],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=stores_to_use))])
                )
                text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
                return (text or "Không thể phân tích ảnh.") + extract_citations(response)
            else:
                # Dùng LangChain nếu không cần RAG
                msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"}}])
                return VISION_LLM.invoke([msg]).content
        except Exception as e:
            return f"Lỗi xử lý ảnh: {e}"

    def _format_history_input(input_dict):
        # Placeholder để pass check input
        return input_dict.get("question", "")

    def get_history(sid, uid):
        return get_session_history(sid, uid)

    return RunnableWithMessageHistory(
        RunnableLambda(route_vision),
        get_history_for_langchain,
        input_messages_key="question",
        input_messages_key_fx=RunnableLambda(_format_history_input),
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID", is_shared=True),
            ConfigurableFieldSpec(id="session_id", annotation=str, name="Session ID", is_shared=True),
        ]
    )


# ==============================================================================
# SECTION 4: KHỞI TẠO CHAIN TOÀN CỤC (ĐỂ API SỬ DỤNG)
# ==============================================================================

# Thay RAG_CHAIN_WITH_HISTORY bằng Agent Executor
RAG_AGENT_EXECUTOR = create_agent_executor(TEXT_LLM)
VISION_CHAIN_WITH_HISTORY = create_vision_chain(VISION_LLM)


# ==============================================================================
# SECTION 5: CÁC HÀM XỬ LÝ CLI
# ==============================================================================

def handle_text_query(query_text, user_id, session_id="default_session"):
    print("--- 🤖 Đang xử lý câu hỏi bằng AI Agent ---")
    agent = RAG_AGENT_EXECUTOR
    if agent is None:
        return

    config_ = {"configurable": {"session_id": session_id, "user_id": user_id}}

    try:
        # AgentExecutor trả về dict, ta dùng ainvoke hoặc invoke
        # Lưu ý: Trong CLI ta dùng invoke đồng bộ
        result = agent.invoke({"question": query_text}, config=config_)
        full_response = result.get("output", "Không có phản hồi.")

        print(f"\n💡 Trả lời:\n{full_response}\n")
        save_session_message(session_id, user_id, query_text, full_response)
    except Exception as e:
        print(f"\nLỗi khi Agent xử lý: {e}")


def handle_multimodal_query(query_text, image_path, user_id, session_id="default_session"):
    print(f"--- 🖼️ Xử lý câu hỏi có ảnh: {os.path.basename(image_path)} ---")
    chain = VISION_CHAIN_WITH_HISTORY
    if chain is None:
        return

    input_data = {"question": query_text, "image_path": image_path}
    config_ = {"configurable": {"session_id": session_id, "user_id": user_id}}

    try:
        # Vision chain trả về string trực tiếp
        full_response = chain.invoke(input_data, config=config_)
        print(f"\n💡 Trả lời:\n{full_response}\n")
        save_session_message(session_id, user_id, query_text, full_response, image_path=image_path)
    except Exception as e:
        print(f"\nLỗi khi xử lý câu hỏi ảnh: {e}")


def handle_pdf_upload(pdf_path: str, session_id: str, user_id: str):
    print(f"\n⏳ Đang xử lý file: {pdf_path}...")
    try:
        file_id = save_pdf_to_mongo(pdf_path, session_id, user_id)
        if not file_id: return
        doc = DB_DOCUMENTS_COLLECTION.find_one({"_id": ObjectId(file_id)})
        if not doc: return

        if doc.get("status") == "processed":
            print("File đã được xử lý trước đó.")
        else:
            process_and_vectorize_pdf(pdf_path, session_id, str(doc["_id"]))
            print("Xử lý và tạo File Store riêng thành công.")
    except Exception as ex:
        print(f"Lỗi nghiêm trọng khi xử lý file PDF: {ex}")


# ==============================================================================
# SECTION 6: MAIN CLI
# ==============================================================================

def main():
    print("🤖 Chatbot CUSC (Agent + Google File Search) sẵn sàng!")
    print("=" * 30)
    print("[1] Tạo session mới")
    print("[2] Tiếp tục session cũ")

    user_id = "6915f6a4d74b46caa1d4d0b2"
    choice = input("Lựa chọn của bạn (1 hoặc 2): ").strip()

    if choice == '2':
        sessions = list_sessions(limit=10, user_id=user_id)
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
            handle_multimodal_query(user_input, img_path, user_id, session_id)
        else:
            handle_text_query(user_input, user_id, session_id)


if __name__ == "__main__":
    main()
