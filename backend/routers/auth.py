"""
Authentication Router
Handles user registration, login, and token management
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr

from backend.models.user import (
    UserCreate, UserLogin, UserResponse, Token
)
from backend.services.user_service import create_user, authenticate_user, verify_user, get_user_by_email
from backend.services.auth_service import create_access_token
from backend.services.email_service import (
    generate_verification_token, decode_verification_token, send_verification_email
)
from backend.dependencies import get_current_user


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
    description="Authenticate with email and password to receive JWT token"
)
async def login(credentials: UserLogin):
    """
    Login with email and password.
    
    Returns a JWT access token that should be included in the Authorization header
    for authenticated requests: `Authorization: Bearer <token>`
    """
    user = authenticate_user(credentials.email, credentials.password)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Refuse logins for accounts that haven't completed email verification.
    # We deliberately check this AFTER verifying the password so this branch
    # cannot be used to enumerate which emails are registered.
    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Tài khoản chưa được xác thực email. "
                "Vui lòng kiểm tra hộp thư hoặc yêu cầu gửi lại email xác thực."
            ),
            headers={"X-Auth-Error-Code": "email_not_verified"},
        )

    access_token, expires_in = create_access_token(user_id=str(user["_id"]))
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in
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
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout (client-side)",
    description="Logout endpoint (JWT tokens are stateless, client should discard token)"
)
async def logout():
    """
    Logout endpoint.
    
    Note: JWT tokens are stateless. The client should discard the token.
    This endpoint is provided for API completeness.
    """
    return {"message": "Successfully logged out. Please discard your token."}


@router.delete(
    "/account",
    status_code=status.HTTP_200_OK,
    summary="Delete user account",
    description="Permanently delete the current user's account"
)
async def delete_account(current_user: dict = Depends(get_current_user)):
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
            detail="Không thể xóa tài khoản. Vui lòng thử lại sau."
        )
    
    return {"message": "Tài khoản đã được xóa thành công."}


