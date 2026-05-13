"""
Cookie helpers for auth.

Centralises the flags we use for the access- and refresh-token cookies so
every endpoint sets / clears them the same way. The cookie names + flags
are pulled from `chatbot.config.config` (driven by env) so they can be
tuned per environment without touching application code.
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import Response

from chatbot.config import config as app_config


def _common_cookie_kwargs() -> dict:
    """Shared kwargs for both set_cookie and delete_cookie."""
    kwargs: dict = {
        "httponly": True,
        "secure": app_config.COOKIE_SECURE,
        "samesite": app_config.COOKIE_SAMESITE,
        "path": "/",
    }
    if app_config.COOKIE_DOMAIN:
        kwargs["domain"] = app_config.COOKIE_DOMAIN
    return kwargs


def set_access_token_cookie(response: Response, token: str, max_age_seconds: int) -> None:
    response.set_cookie(
        key=app_config.ACCESS_TOKEN_COOKIE,
        value=token,
        max_age=max_age_seconds,
        **_common_cookie_kwargs(),
    )


def set_refresh_token_cookie(response: Response, token: str) -> None:
    max_age = int(timedelta(days=app_config.REFRESH_TOKEN_EXPIRE_DAYS).total_seconds())
    response.set_cookie(
        key=app_config.REFRESH_TOKEN_COOKIE,
        value=token,
        max_age=max_age,
        **_common_cookie_kwargs(),
    )


def clear_auth_cookies(response: Response) -> None:
    """Clear both access- and refresh-token cookies on logout."""
    delete_kwargs: dict = {"path": "/"}
    if app_config.COOKIE_DOMAIN:
        delete_kwargs["domain"] = app_config.COOKIE_DOMAIN
    response.delete_cookie(key=app_config.ACCESS_TOKEN_COOKIE, **delete_kwargs)
    response.delete_cookie(key=app_config.REFRESH_TOKEN_COOKIE, **delete_kwargs)
