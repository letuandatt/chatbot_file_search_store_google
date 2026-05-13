"""
API Dependencies
Dependency injection for FastAPI endpoints
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.services.auth_service import decode_access_token
from backend.services.user_service import get_user_by_id


# Security scheme for JWT Bearer token
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    Dependency to get the current authenticated user from JWT token
    Raises HTTPException if not authenticated
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
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
            detail="User account is deactivated"
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    Dependency to optionally get the current user (for endpoints that work with or without auth)
    Returns None if not authenticated
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# AppContainer singleton for the chatbot agent. The container itself
# owns the singleton + lock; this dependency just exposes it to FastAPI.
def get_app_container():
    """
    Return the process-wide chatbot AppContainer.

    The first call builds the container (GenAI client, agent executor,
    memory service, file watcher); subsequent calls return the same
    instance. Construction is thread-safe.
    """
    from chatbot.app_container import AppContainer
    return AppContainer.instance()
