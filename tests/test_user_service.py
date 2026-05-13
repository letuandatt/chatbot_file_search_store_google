"""
Unit tests for `backend.services.user_service`.

These tests exercise the user CRUD paths against a mongomock-backed
collection (see `tests.conftest.mongo_db`). They cover:

  * create_user happy path + duplicate-email rejection,
  * authenticate_user wrong-password / missing-user / inactive-user
    cases,
  * verify_user — including the Bug 7 regression: must return False
    when the user has been deleted between sending the verification
    email and clicking the link,
  * update_user — covers the Bug 1 regression: must return the
    UPDATED document, not the stale one,
  * delete_user.
"""

from __future__ import annotations

from backend.services import user_service as us


def test_create_user_persists_lowercase_email(mongo_db):
    user = us.create_user(email="Test@Example.COM", password="hunter2", full_name="Alice")
    assert user is not None
    assert user["email"] == "test@example.com"
    assert user["is_verified"] is False
    assert user["is_active"] is True
    # The plain password must never be stored.
    assert "password" not in user
    assert user["hashed_password"] != "hunter2"


def test_create_user_rejects_duplicate_email(mongo_db):
    assert us.create_user(email="dup@example.com", password="hunter2") is not None
    # Same email, different case, should still be rejected because
    # the service lower-cases on both read and write.
    assert us.create_user(email="DUP@example.com", password="hunter2") is None


def test_authenticate_user_happy_path(mongo_db):
    us.create_user(email="login@example.com", password="hunter2")
    user = us.authenticate_user(email="login@example.com", password="hunter2")
    assert user is not None
    assert user["email"] == "login@example.com"


def test_authenticate_user_wrong_password(mongo_db):
    us.create_user(email="login@example.com", password="hunter2")
    assert us.authenticate_user(email="login@example.com", password="WRONG") is None


def test_authenticate_user_missing_email(mongo_db):
    assert us.authenticate_user(email="nope@example.com", password="hunter2") is None


def test_authenticate_user_inactive_account(mongo_db):
    us.create_user(email="inactive@example.com", password="hunter2")
    # Manually flip is_active False (we don't have a public setter)
    mongo_db.get_collection("users").update_one(
        {"email": "inactive@example.com"},
        {"$set": {"is_active": False}},
    )
    assert us.authenticate_user(email="inactive@example.com", password="hunter2") is None


def test_verify_user_marks_existing_user(mongo_db):
    user = us.create_user(email="verify@example.com", password="hunter2")
    assert user is not None
    assert user["is_verified"] is False

    ok = us.verify_user(str(user["_id"]))
    assert ok is True

    fresh = us.get_user_by_id(str(user["_id"]))
    assert fresh["is_verified"] is True


def test_verify_user_returns_false_for_deleted_user(mongo_db):
    """
    Regression for Bug 7. A verification link clicked after the user
    has been deleted must return False instead of silently "verifying"
    a non-existent user (which used to happen because `update_one`
    with no match is a no-op rather than an error).

    NOTE: pre-PR-#6 behaviour is `verify_user` returning True (or
    False because modified_count == 0); after PR #6 it must be False.
    We assert the post-fix behaviour and let this test fail on master
    until the bug-fix PR lands.
    """
    fake_id = "507f1f77bcf86cd799439011"  # valid ObjectId, no user
    result = us.verify_user(fake_id)
    # Accept both pre-fix (False because nothing modified) and post-fix
    # (False because we explicitly checked existence). The bug was
    # that update_one *would* succeed on an upsert path; our schema
    # doesn't actually upsert, so the bug surfaced only when checked
    # by the calling code. The semantic guarantee we lock in is:
    # never True for a non-existent user.
    assert result is False


def test_update_user_returns_updated_document(mongo_db):
    """
    Regression for Bug 1: `return_document=True` happens to map to
    ReturnDocument.AFTER on current pymongo, but explicit enum makes
    the intent obvious. The functional guarantee here is just that
    callers see the NEW value, not the stale one.
    """
    user = us.create_user(email="update@example.com", password="hunter2", full_name="Old")
    assert user is not None

    updated = us.update_user(str(user["_id"]), {"full_name": "New"})
    assert updated is not None
    assert updated["full_name"] == "New", (
        "update_user must return the post-update document, not the original snapshot"
    )


def test_delete_user(mongo_db):
    user = us.create_user(email="del@example.com", password="hunter2")
    assert user is not None
    user_id = str(user["_id"])

    assert us.delete_user(user_id) is True
    assert us.get_user_by_id(user_id) is None
    # Deleting again should report False (nothing to delete).
    assert us.delete_user(user_id) is False
