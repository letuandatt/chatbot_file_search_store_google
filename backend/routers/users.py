"""
Users Router
Handles user profile management
"""
from fastapi import APIRouter, Depends, HTTPException, status

from backend.models.user import (
    UserResponse, UserUpdate, ChangePassword
)
from backend.dependencies import get_current_user
from backend.services.user_service import (
    update_user, change_password, delete_user, deactivate_user
)


router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Get the profile of the currently authenticated user"
)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Get the current user's profile information.
    
    Requires authentication via Bearer token.
    """
    return UserResponse(
        id=str(current_user["_id"]),
        email=current_user["email"],
        full_name=current_user.get("full_name"),
        avatar_url=current_user.get("avatar_url"),
        created_at=current_user["created_at"],
        is_active=current_user.get("is_active", True)
    )


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Update the profile of the currently authenticated user"
)
async def update_me(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update the current user's profile.
    
    - **full_name**: Update display name
    - **avatar_url**: Update avatar URL
    """
    updated_user = update_user(
        user_id=str(current_user["_id"]),
        update_data=update_data.model_dump(exclude_unset=True)
    )
    
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )
    
    return UserResponse(
        id=str(updated_user["_id"]),
        email=updated_user["email"],
        full_name=updated_user.get("full_name"),
        avatar_url=updated_user.get("avatar_url"),
        created_at=updated_user["created_at"],
        is_active=updated_user.get("is_active", True)
    )


@router.post(
    "/me/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change password",
    description="Change the current user's password"
)
async def change_user_password(
    password_data: ChangePassword,
    current_user: dict = Depends(get_current_user)
):
    """
    Change the current user's password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (minimum 6 characters)
    """
    success = change_password(
        user_id=str(current_user["_id"]),
        current_password=password_data.current_password,
        new_password=password_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return {"message": "Password changed successfully"}


@router.delete(
    "/me",
    status_code=status.HTTP_200_OK,
    summary="Delete account",
    description="Permanently delete the current user's account",
)
async def delete_me(current_user: dict = Depends(get_current_user)):
    """
    Permanently delete the current user's account.

    **Warning**: This action cannot be undone. All user data will be deleted.

    Returns 200 with a confirmation body. We deliberately do NOT use
    204 No Content here because the error path raises an HTTPException
    with a JSON body, and a 204 response with a body violates the HTTP
    spec (some proxies will strip the body, breaking error reporting).
    """
    success = delete_user(str(current_user["_id"]))

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account",
        )

    return {"message": "Account deleted successfully"}


@router.post(
    "/me/deactivate",
    status_code=status.HTTP_200_OK,
    summary="Deactivate account",
    description="Soft delete - deactivate the current user's account"
)
async def deactivate_me(current_user: dict = Depends(get_current_user)):
    """
    Deactivate the current user's account (soft delete).
    
    The account can potentially be reactivated by an admin.
    """
    success = deactivate_user(str(current_user["_id"]))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate account"
        )
    
    return {"message": "Account deactivated successfully"}
