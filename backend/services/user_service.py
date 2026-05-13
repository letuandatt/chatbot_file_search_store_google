"""
User Service
Handles user CRUD operations
"""
from datetime import datetime, timezone
from typing import Optional
from bson import ObjectId
from pymongo import ReturnDocument

from chatbot.core.db import DB_USERS_COLLECTION, get_mongo_collection
from backend.services.auth_service import hash_password, verify_password


def _now() -> datetime:
    """Single source of truth for timestamps: aware UTC datetime."""
    return datetime.now(timezone.utc)


def create_user(email: str, password: str, full_name: Optional[str] = None) -> Optional[dict]:
    """
    Create a new user in the database
    Returns the created user document or None if email already exists
    """
    if DB_USERS_COLLECTION is None:
        print("[user_service] Users collection not initialized")
        return None
    
    # Check if email already exists
    existing = DB_USERS_COLLECTION.find_one({"email": email.lower()})
    if existing:
        return None  # Email already registered
    
    user_doc = {
        "email": email.lower(),
        "hashed_password": hash_password(password),
        "full_name": full_name,
        "avatar_url": None,
        "created_at": _now(),
        "updated_at": _now(),
        "is_active": True,
        "is_verified": False,  # For future email verification
        "preferences": {}  # User preferences/settings
    }
    
    try:
        result = DB_USERS_COLLECTION.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc
    except Exception as e:
        print(f"[user_service] Error creating user: {e}")
        return None


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Authenticate a user by email and password
    Returns user document if valid, None otherwise
    """
    if DB_USERS_COLLECTION is None:
        return None
    
    user = DB_USERS_COLLECTION.find_one({"email": email.lower()})
    if not user:
        return None
    
    if not user.get("is_active", True):
        return None  # Account is deactivated
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    return user


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get a user by their ObjectId string"""
    if DB_USERS_COLLECTION is None:
        return None
    
    try:
        return DB_USERS_COLLECTION.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def get_user_by_email(email: str) -> Optional[dict]:
    """Get a user by their email address"""
    if DB_USERS_COLLECTION is None:
        return None
    
    return DB_USERS_COLLECTION.find_one({"email": email.lower()})


def update_user(user_id: str, update_data: dict) -> Optional[dict]:
    """
    Update user profile
    Returns updated user document or None
    """
    if DB_USERS_COLLECTION is None:
        return None
    
    # Filter out None values and add updated_at
    update_fields = {k: v for k, v in update_data.items() if v is not None}
    update_fields["updated_at"] = _now()
    
    try:
        # `return_document=True` happens to map to ReturnDocument.AFTER in
        # current pymongo versions, but that's an undocumented coincidence.
        # The enum makes the intent explicit and future-proof.
        result = DB_USERS_COLLECTION.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        return result
    except Exception as e:
        print(f"[user_service] Error updating user: {e}")
        return None


def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """
    Change user password
    Returns True if successful, False otherwise
    """
    if DB_USERS_COLLECTION is None:
        return False
    
    user = get_user_by_id(user_id)
    if not user:
        return False
    
    # Verify current password
    if not verify_password(current_password, user["hashed_password"]):
        return False
    
    # Update to new password
    try:
        DB_USERS_COLLECTION.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "hashed_password": hash_password(new_password),
                    "updated_at": _now()
                }
            }
        )
        return True
    except Exception as e:
        print(f"[user_service] Error changing password: {e}")
        return False


def delete_user(user_id: str) -> bool:
    """
    Delete a user account (hard delete)
    Returns True if successful
    """
    if DB_USERS_COLLECTION is None:
        return False
    
    try:
        result = DB_USERS_COLLECTION.delete_one({"_id": ObjectId(user_id)})
        
        # Also delete user's sessions
        sessions_coll = get_mongo_collection("sessions")
        if sessions_coll:
            sessions_coll.delete_many({"user_id": user_id})
        
        return result.deleted_count > 0
    except Exception as e:
        print(f"[user_service] Error deleting user: {e}")
        return False


def deactivate_user(user_id: str) -> bool:
    """
    Soft delete - deactivate a user account
    Returns True if successful
    """
    if DB_USERS_COLLECTION is None:
        return False
    
    try:
        result = DB_USERS_COLLECTION.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_active": False, "updated_at": _now()}}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"[user_service] Error deactivating user: {e}")
        return False


def verify_user(user_id: str) -> bool:
    """
    Mark a user's email as verified.

    Returns True only when an existing user document was updated. A stale
    verification token whose user has since been deleted will return False
    (previously the code happily "verified" a non-existent user because
    update_one + an upsert-less filter silently no-ops).
    """
    if DB_USERS_COLLECTION is None:
        return False

    try:
        oid = ObjectId(user_id)
    except Exception:
        return False

    try:
        existing = DB_USERS_COLLECTION.find_one({"_id": oid}, {"_id": 1})
        if existing is None:
            return False

        result = DB_USERS_COLLECTION.update_one(
            {"_id": oid},
            {"$set": {"is_verified": True, "updated_at": _now()}},
        )
        # modified_count == 0 means the user was already verified, but the
        # operation still succeeded conceptually.
        return result.matched_count > 0
    except Exception as e:
        print(f"[user_service] Error verifying user: {e}")
        return False


def is_user_verified(user_id: str) -> bool:
    """
    Check if a user's email is verified.
    """
    user = get_user_by_id(user_id)
    if not user:
        return False
    return user.get("is_verified", False)
