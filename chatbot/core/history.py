from datetime import datetime, timezone
from chatbot.core.db import get_mongo_collection
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from pymongo import DESCENDING

MAX_HISTORY = 20

def save_session_message(
    session_id: str, 
    user_id: str, 
    question: str, 
    answer: str, 
    image_gridfs_id: str | None = None,
    thinking_steps: list | None = None
):
    coll = get_mongo_collection("sessions")
    if coll is None:
        print("[core.history] sessions collection missing.")
        return
    # Store an aware UTC timestamp instead of a naive local-time ISO
    # string. The previous value (`datetime.now().isoformat()`) silently
    # drifted with the host's timezone — sessions created on a UTC
    # container compared against sessions created on a +07:00 dev box
    # would sort wrong by `updated_at`.
    now = datetime.now(timezone.utc)
    message_data = {
        "question": question,
        "answer": answer,
        "image_gridfs_id": image_gridfs_id,
        "thinking_steps": thinking_steps,
        "timestamp": now,
    }
    try:
        coll.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message_data},
                "$set": {"updated_at": now},
                "$setOnInsert": {"user_id": user_id, "created_at": now}
            },
            upsert=True
        )
    except Exception as e:
        print(f"[core.history.save_session_message] {e}")
        try:
            coll.update_one({"session_id": session_id}, {"$push": {"messages": message_data}, "$set": {"updated_at": now}}, upsert=True)
        except Exception as ex:
            print(f"[core.history.save_session_message fallback] {ex}")

def load_session_messages(session_id: str, user_id: str, limit: int = 25):
    coll = get_mongo_collection("sessions")
    if coll is None:
        return InMemoryChatMessageHistory()
    session = coll.find_one({"session_id": session_id, "user_id": user_id})
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


def get_history_for_langchain(session_id: str, user_id: str):
    return load_session_messages(session_id, user_id)


get_session_history = get_history_for_langchain

