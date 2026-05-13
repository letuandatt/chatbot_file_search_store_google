"""
Authentication Router
Handles user registration, login, and token management
"""
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr

from backend.dependencies import get_current_user
from backend.models.user import Token, UserCreate, UserLogin, UserResponse
from backend.services.auth_service import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
)
from backend.services.cookie_service import (
    clear_auth_cookies,
    set_access_token_cookie,
    set_refresh_token_cookie,
)
from backend.services.email_service import (
    decode_verification_token,
    generate_verification_token,
    send_verification_email,
)
from backend.services.user_service import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
    verify_user,
)
from chatbot.config import config as app_config


router = APIRouter(prefix="/auth", tags=["Authentication"])


class ResendVerificationRequest(BaseModel):
    """Schema for resend verification request"""
    email: EmailStr


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account and send verification email"
)
async def register(user_data: UserCreate):
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **password**: Password (minimum 6 characters)
    - **full_name**: Optional full name
    
    A verification email will be sent to the provided email address.
    """
    user = create_user(
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name
    )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Send verification email
    token = generate_verification_token(str(user["_id"]))
    email_sent = send_verification_email(
        to_email=user["email"],
        user_name=user.get("full_name"),
        verification_token=token
    )
    
    if not email_sent:
        print(f"[auth] Warning: Could not send verification email to {user['email']}")
    
    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user.get("full_name"),
        avatar_url=user.get("avatar_url"),
        created_at=user["created_at"],
        is_active=user.get("is_active", True)
    )


@router.post(
    "/login",
    response_model=Token,
    summary="Login to get access token",
    description="Authenticate with email and password to receive JWT token",
)
async def login(credentials: UserLogin, response: Response):
    """
    Login with email and password.

    On success the access + refresh tokens are stored in httpOnly,
    Secure (in production), SameSite=Lax cookies so the browser cannot
    expose them to JavaScript (eliminating the XSS → token-theft path).

    For API clients that don't run in a browser the access token is
    still returned in the response body, and the `Authorization: Bearer`
    header continues to be accepted by all protected endpoints.
    """
    user = authenticate_user(credentials.email, credentials.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user["_id"])
    access_token, expires_in = create_access_token(user_id=user_id)
    refresh_token = create_refresh_token(user_id)

    set_access_token_cookie(response, access_token, expires_in)
    set_refresh_token_cookie(response, refresh_token)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.get(
    "/verify",
    summary="Verify email address",
    description="Verify user's email address using the token from verification email"
)
async def verify_email(token: str):
    """
    Verify email address using the token sent via email.
    
    - **token**: Verification token from the email link
    """
    user_id = decode_verification_token(token)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token không hợp lệ hoặc đã hết hạn. Vui lòng yêu cầu gửi lại email xác thực."
        )
    
    success = verify_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể xác thực email. User không tồn tại hoặc đã được xác thực trước đó."
        )
    
    return {
        "message": "Email đã được xác thực thành công! Bạn có thể đăng nhập ngay.",
        "verified": True
    }


@router.post(
    "/resend-verification",
    summary="Resend verification email",
    description="Resend the verification email to a user"
)
async def resend_verification(request: ResendVerificationRequest):
    """
    Resend verification email.
    
    - **email**: Email address to send verification to
    """
    user = get_user_by_email(request.email)
    
    if user is None:
        # Don't reveal if email exists for security
        return {"message": "Nếu email tồn tại trong hệ thống, bạn sẽ nhận được email xác thực."}
    
    if user.get("is_verified", False):
        return {"message": "Email này đã được xác thực trước đó."}
    
    # Send verification email
    token = generate_verification_token(str(user["_id"]))
    email_sent = send_verification_email(
        to_email=user["email"],
        user_name=user.get("full_name"),
        verification_token=token
    )
    
    if not email_sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể gửi email xác thực. Vui lòng thử lại sau."
        )
    
    return {"message": "Email xác thực đã được gửi. Vui lòng kiểm tra hộp thư của bạn."}


@router.post(
    "/refresh",
    response_model=Token,
    summary="Mint a new access token using the refresh-token cookie",
    description="Refresh the short-lived access token using the long-lived refresh token cookie.",
)
async def refresh_token_endpoint(
    response: Response,
    refresh_cookie: Optional[str] = Cookie(default=None, alias=app_config.REFRESH_TOKEN_COOKIE),
):
    """
    Exchange a valid refresh-token cookie for a fresh access token.

    Only the refresh cookie is trusted here — we deliberately do NOT
    accept the refresh token in the request body or Authorization
    header, because httpOnly is the property that protects it.
    """
    if not refresh_cookie:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )

    user_id = decode_refresh_token(refresh_cookie)
    if user_id is None:
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user = get_user_by_id(user_id)
    if user is None or not user.get("is_active", True):
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account not found or deactivated",
        )

    access_token, expires_in = create_access_token(user_id=user_id)
    set_access_token_cookie(response, access_token, expires_in)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout",
    description="Clear the auth cookies. JWT itself is stateless, so the access token remains valid until it expires; clients should rely on the cookies being cleared.",
)
async def logout(response: Response):
    """
    Clear the access + refresh httpOnly cookies.

    JWTs are stateless and cannot be invalidated server-side without a
    revocation list (deferred to a follow-up). Clearing the cookies is
    enough to prevent the browser from re-authenticating on its own.
    """
    clear_auth_cookies(response)
    return {"message": "Successfully logged out."}


@router.delete(
    "/account",
    status_code=status.HTTP_200_OK,
    summary="Delete user account",
    description="Permanently delete the current user's account",
)
async def delete_account(
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete the current user's account permanently.

    This action cannot be undone. All user data including sessions will be deleted.
    """
    from backend.services.user_service import delete_user

    user_id = str(current_user["_id"])
    success = delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể xóa tài khoản. Vui lòng thử lại sau.",
        )

    clear_auth_cookies(response)
    return {"message": "Tài khoản đã được xóa thành công."}


