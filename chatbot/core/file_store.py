import os
import uuid
from datetime import datetime, timezone
from bson.objectid import ObjectId
from chatbot.core.db import DB_DOCUMENTS_COLLECTION, FS
from chatbot.core.utils import compute_file_hash

def save_pdf_to_mongo(file_path: str, session_id: str, user_id: str, original_filename: str = None) -> str | None:
    """
    Save PDF into GridFS + documents collection. Return documents._id as string.
    Deduplicate by file_hash per user.
    """
    fs_client = FS
    coll = DB_DOCUMENTS_COLLECTION
    if fs_client is None or coll is None:
        print("[core.file_store] DB or FS not ready.")
        return None
    try:
        file_hash = compute_file_hash(file_path)
        # Use original filename if provided, otherwise fallback to basename
        file_name = original_filename or os.path.basename(file_path)
        # check if same file already uploaded in this session
        existing = coll.find_one({"file_hash": file_hash, "user_id": user_id, "session_id": session_id})
        if existing:
            return str(existing["_id"])
        # reuse per-user same hash if exists
        hash_existing = coll.find_one({"file_hash": file_hash, "user_id": user_id})
        if hash_existing:
            file_gridfs_id = hash_existing["file_gridfs_id"]
        else:
            with open(file_path, "rb") as f:
                file_id = fs_client.put(f, filename=file_name, metadata={"original_user": user_id})
            file_gridfs_id = str(file_id)
        result = coll.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "filename": file_name,
            "file_gridfs_id": file_gridfs_id,
            "file_hash": file_hash,
            # Aware UTC datetime keeps timestamps comparable across the
            # codebase (auth_service uses datetime.now(timezone.utc) too).
            # Mongo stores it as a BSON Date, which is preferable to a
            # naive ISO string for sort/range queries.
            "created_at": datetime.now(timezone.utc),
            "status": "uploaded"
        })
        return str(result.inserted_id)
    except Exception as e:
        print(f"[core.file_store.save_pdf_to_mongo] {e}")
        return None

def process_and_vectorize_pdf(file_path: str, session_id: str, doc_id: str, genai_client):
    """
    Use Google GenAI client to create a file_search_store and upload file.
    On success update doc.status -> processed and set file_store_name.
    genai_client: instance genai.Client
    """
    coll = DB_DOCUMENTS_COLLECTION
    client = genai_client
    if coll is None or client is None:
        print("[core.file_store] DB or genai client not ready.")
        return
    file_name = os.path.basename(file_path)
    try:
        store_display_name = f"session-{session_id[:8]}-file-{doc_id[:8]}-{uuid.uuid4().hex[:8]}"
        file_store = client.file_search_stores.create(config={'display_name': store_display_name})
        client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=file_store.name,
            config={'display_name': file_name}
        )
        coll.update_one({"_id": ObjectId(doc_id)}, {"$set": {"status": "processed", "file_store_name": file_store.name}})
        print(f"[core.file_store] Processed {file_name} -> {file_store.name}")
    except Exception as e:
        print(f"[core.file_store.process_and_vectorize_pdf] {e}")
        try:
            coll.update_one({"_id": ObjectId(doc_id)}, {"$set": {"status": "error_processing", "error": str(e)}})
        except Exception:
            pass

def get_session_file_stores(session_id: str) -> list[str]:
    """
    Return list of file_store_name for the given session (processed files).
    """
    coll = DB_DOCUMENTS_COLLECTION
    if coll is None:
        return []
    try:
        cursor = coll.find({"session_id": session_id, "status": "processed"}, {"file_store_name": 1})
        return [doc.get("file_store_name") for doc in cursor if doc.get("file_store_name")]
    except Exception:
        return []
