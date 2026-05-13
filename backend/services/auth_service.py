from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from passlib.context import CryptContext

from chatbot.config import config as app_config


# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password) -> bool:
    """Verify a plain text password against a hashed password"""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password) -> str:
    """Hash a plain text password using bcrypt"""
    return pwd_context.hash(password)


# Alias for compatibility
get_password_hash = hash_password


# --- JWT Token Handling ---
def create_access_token(data: dict = None, user_id: str = None, expires_delta: Optional[timedelta] = None) -> tuple[str, int]:
    """
    Create a JWT access token.
    
    Args:
        data: Optional dict with data to encode (for compatibility)
        user_id: User ID to encode in token
        expires_delta: Optional custom expiration time
    
    Returns:
        Tuple of (token, expires_in_seconds)
    """
    if data is not None:
        to_encode = data.copy()
    else:
        to_encode = {"sub": user_id}
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
        expires_in = int(expires_delta.total_seconds())
    else:
        expires_in = app_config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        expire = datetime.now(timezone.utc) + timedelta(minutes=app_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        app_config.JWT_SECRET_KEY,
        algorithm=app_config.JWT_ALGORITHM
    )
    
    return encoded_jwt, expires_in


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode and validate a JWT access token.
    Returns user_id if valid, None otherwise.
    """
    try:
        payload = jwt.decode(
            token,
            app_config.JWT_SECRET_KEY,
            algorithms=[app_config.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        return None

    # Refresh tokens must not be accepted on access-token code paths.
    if payload.get("type") == "refresh":
        return None
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id


def create_refresh_token(user_id: str) -> str:
    """
    Create a refresh token (longer expiry) for the given user_id.
    """
    expire = datetime.now(timezone.utc) + timedelta(days=app_config.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
    }

    return jwt.encode(
        payload,
        app_config.JWT_SECRET_KEY,
        algorithm=app_config.JWT_ALGORITHM,
    )


def decode_refresh_token(token: str) -> Optional[str]:
    """
    Decode a refresh token and return the user_id if valid.

    Tokens that lack the `type=refresh` claim are rejected so a leaked
    access token cannot be used in place of a refresh token (and vice
    versa via `decode_access_token`, which only succeeds for tokens
    without an incompatible type claim).
    """
    try:
        payload = jwt.decode(
            token,
            app_config.JWT_SECRET_KEY,
            algorithms=[app_config.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        return None

    if payload.get("type") != "refresh":
        return None
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id
