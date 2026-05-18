"""
Backwards-compatible entry point.

The real CLI lives in `chatbot.cli` and the service container in
`chatbot.app_container`. This module exists so that:

  * `python -m chatbot.main` still launches the REPL, and
  * existing imports such as `from chatbot.main import AppContainer`
    keep working without immediately spinning up a second container
    (the old layout did `APP = AppContainer()` at module level).

Prefer importing `AppContainer` from `chatbot.app_container` and the
CLI helpers from `chatbot.cli` in new code.
"""
from __future__ import annotations

from chatbot.app_container import AppContainer
from chatbot.cli import handle_pdf_upload, handle_unified_query, main

__all__ = [
    "AppContainer",
    "handle_pdf_upload",
    "handle_unified_query",
    "main",
]


if __name__ == "__main__":
    main()
