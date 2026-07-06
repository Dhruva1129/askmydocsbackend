"""
Auth Routes
-----------
POST /auth/register   Create a new user account
POST /auth/login      Login → JWT access token
GET  /auth/me         Get current user profile
PUT  /auth/me         Update own profile (name, password)
"""

from datetime import timezone, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import User
from app.schemas.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.services.auth import (
    authenticate_user,
    create_user,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.core.security import JWT_EXPIRY_HOURS

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=201)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.
    - First user ever created automatically gets the 'admin' role.
    - All subsequent users default to 'employee' unless explicitly set.
    """
    from sqlalchemy import func, select
    count_result = await db.execute(select(func.count()).select_from(User))
    user_count = count_result.scalar()

    role = "admin" if user_count == 0 else request.role
    user = await create_user(
        db,
        email=request.email,
        password=request.password,
        full_name=request.full_name,
        role=role,
        department=request.department,
    )
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and return a JWT token."""
    user = await authenticate_user(db, request.email, request.password)
    token = create_access_token(user)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=JWT_EXPIRY_HOURS * 3600,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return current_user


@router.put("/me/password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change the current user's password."""
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    current_user.hashed_password = hash_password(new_password)
    await db.commit()
    return {"message": "Password updated successfully"}
