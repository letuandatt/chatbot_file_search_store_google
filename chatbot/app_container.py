"""
AppContainer — process-wide service container for the LexMind chatbot.

This module owns the construction of the heavy, long-lived services
(GenAI client, vision service, agent executor, memory service, file
watcher). It deliberately does NOT instantiate anything at import
time — both the FastAPI backend (`backend.dependencies.get_app_container`)
and the CLI (`chatbot.cli.main`) are responsible for creating exactly
one `AppContainer` per process via `AppContainer.instance()`.

The previous layout placed `APP = AppContainer()` at module level in
`chatbot/main.py`, which meant simply importing the CLI module from
the API spun up a second agent (with its own watcher, GenAI client,
etc.) alongside the lazy one the API created. Centralising the
singleton here removes that duplication.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import google.genai as genai

from chatbot.config import config as app_config
from chatbot.core.db import init_db
from chatbot.core.memory_profile import build_user_memory
from chatbot.core.watcher import app_watcher
from chatbot.router.dispatcher import build_rag_agent
from chatbot.services.vision_service import VisionService

logger = logging.getLogger(__name__)


class AppContainer:
    """Holds long-lived chatbot services. Treat as a process singleton."""

    # Class-level state for the singleton accessor. Instances created
    # directly (e.g. in a unit test) bypass this and are NOT tracked.
    _instance: Optional["AppContainer"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        logger.info("[AppContainer] initialising services...")
        init_db()

        try:
            self.genai_client = genai.Client(api_key=app_config.GOOGLE_API_KEY)
            logger.info("[AppContainer] GenAI client initialised")
        except Exception as exc:  # pragma: no cover - depends on external key
            logger.error("[AppContainer] GenAI client init failed: %s", exc)
            self.genai_client = None

        self.vision_service = VisionService(self.genai_client)

        if self.genai_client is not None:
            self.agent_executor, self.text_llm = build_rag_agent(
                self.genai_client, self.vision_service
            )
            self.memory_service = build_user_memory(self.text_llm)
        else:
            self.agent_executor = None
            self.text_llm = None
            self.memory_service = None

        # The watcher polls Mongo for newly-uploaded PDFs. It must be
        # running for /chat/upload to ever transition a file from
        # "uploaded" to "processed".
        app_watcher.start()
        logger.info("[AppContainer] file watcher started")

    def shutdown(self) -> None:
        """Stop background threads owned by the container."""
        try:
            app_watcher.stop()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[AppContainer] watcher stop failed: %s", exc)

    @classmethod
    def instance(cls) -> "AppContainer":
        """Return the process-wide singleton, building it on first use."""
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                logger.info("[AppContainer] creating singleton instance")
                cls._instance = cls()
            else:
                logger.debug("[AppContainer] singleton already created during contention")
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the singleton reference. Tests only."""
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance.shutdown()
                finally:
                    cls._instance = None
