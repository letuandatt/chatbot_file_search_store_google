"""
Pytest fixtures for LexMindChatbot.

Two pieces of import-time gymnastics happen here:

  1. We populate dummy environment variables BEFORE the chatbot.config
     module is loaded, so `app_config.JWT_SECRET_KEY`, `MONGO_URI`,
     and friends are present. The app reads them at import time
     (via python-dotenv).

  2. We stub out `google.genai` BEFORE `chatbot.core.watcher` is
     imported, so simply importing the backend does NOT try to build
     a real GenAI client (which would either hang, hit the network,
     or raise depending on the environment).

After that, each test patches the MongoDB collections in
`chatbot.core.db` with `mongomock` collections via the `mongo_db`
fixture, giving us a real-PyMongo-API in-memory database to assert
against.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Iterator

# --- 1. Env vars must be set BEFORE chatbot.config is imported ---
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/_unused_in_tests")
os.environ.setdefault("GOOGLE_API_KEY", "test-genai-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("LAW_MAIN_STORE_NAME", "test-store")
os.environ.setdefault("GMAIL_USER", "test@example.com")
os.environ.setdefault("GMAIL_APP_PASSWORD", "test-app-password")
os.environ.setdefault("FRONTEND_URL", "http://localhost:3000")

# --- 2. Stub google.genai before chatbot.core.watcher gets imported ---
_fake_genai = types.ModuleType("google.genai")


class _FakeGenaiClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        # Minimal attribute access for callers that try to inspect
        # sub-services. They'll just see None and skip work.
        self.file_search_stores = None


_fake_genai.Client = _FakeGenaiClient
_fake_google = types.ModuleType("google")
_fake_google.genai = _fake_genai
sys.modules.setdefault("google", _fake_google)
sys.modules["google.genai"] = _fake_genai

import mongomock  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def mongo_db(monkeypatch) -> Iterator[mongomock.Database]:
    """
    Provide a fresh in-memory MongoDB for the test and swap it into
    `chatbot.core.db` plus the per-service module-level references.

    The real `init_db()` is also patched to a no-op so any code path
    that re-initialises the database during the test doesn't reach
    the network.
    """
    from chatbot.core import db as core_db

    client = mongomock.MongoClient()
    database = client["test_db"]

    sessions = database.get_collection("sessions")
    documents = database.get_collection("documents")
    users = database.get_collection("users")

    # Patch the module-level globals that services read directly.
    monkeypatch.setattr(core_db, "_mongo_client", client, raising=False)
    monkeypatch.setattr(core_db, "_mongo_db", database, raising=False)
    monkeypatch.setattr(core_db, "DB_COLLECTION", sessions, raising=False)
    monkeypatch.setattr(core_db, "DB_DOCUMENTS_COLLECTION", documents, raising=False)
    monkeypatch.setattr(core_db, "DB_USERS_COLLECTION", users, raising=False)
    # GridFS is harder to mock convincingly; tests that don't touch
    # file uploads can leave it None.
    monkeypatch.setattr(core_db, "FS", None, raising=False)
    monkeypatch.setattr(core_db, "init_db", lambda: None)

    # The user_service module imports DB_USERS_COLLECTION by name at
    # import time, so we need to patch THAT reference too.
    from backend.services import user_service as us

    monkeypatch.setattr(us, "DB_USERS_COLLECTION", users, raising=False)

    yield database
