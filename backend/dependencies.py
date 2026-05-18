"""
API Dependencies
Dependency injection for FastAPI endpoints
"""
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.services.auth_service import decode_access_token
from backend.services.user_service import get_user_by_id
from chatbot.config import config as app_config


# Security scheme for JWT Bearer token. `auto_error=False` so the cookie
# fallback below can run when no Authorization header is sent.
security = HTTPBearer(auto_error=False)


def _extract_token(
    credentials: Optional[HTTPAuthorizationCredentials],
    cookie_token: Optional[str],
) -> Optional[str]:
    """Pick the access token from the cookie first, then the Authorization header.

    Cookie wins so a stale `Bearer` header (e.g. from a forgotten dev tool
    tab) cannot override the freshly issued cookie. The header path is
    kept for API clients (curl, tests, mobile) that don't use cookies.
    """
    if cookie_token:
        return cookie_token
    if credentials is not None and credentials.credentials:
        return credentials.credentials
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token_cookie: Optional[str] = Cookie(default=None, alias=app_config.ACCESS_TOKEN_COOKIE),
) -> dict:
    """
    Dependency to get the current authenticated user from JWT token.

    Accepts the token from either:
      * the `access_token` httpOnly cookie (preferred — frontend path), or
      * the `Authorization: Bearer <token>` header (API clients).

    Raises HTTPException(401) if neither is present or valid.
    """
    token = _extract_token(credentials, access_token_cookie)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token_cookie: Optional[str] = Cookie(default=None, alias=app_config.ACCESS_TOKEN_COOKIE),
) -> Optional[dict]:
    """
    Dependency to optionally get the current user (for endpoints that work
    with or without auth). Returns None if not authenticated.
    """
    if _extract_token(credentials, access_token_cookie) is None:
        return None

    try:
        return await get_current_user(credentials, access_token_cookie)
    except HTTPException:
        return None


# AppContainer singleton for chatbot agent
_app_container = None
_app_container_lock = None


def get_app_container():
    """
    Dependency to get the AppContainer singleton
    Lazy initialization to avoid loading heavy models until needed
    """
    global _app_container
    
    if _app_container is None:
        from chatbot.main import AppContainer
        _app_container = AppContainer()
    
    return _app_container
