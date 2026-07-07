"""
Authentication Service
-----------------------
- JWT creation and verification (python-jose)
- Password hashing and verification (passlib/bcrypt)
- FastAPI dependency: get_current_user
- Role enforcement dependency factory: require_role()
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import JWT_ALGORITHM, JWT_EXPIRY_HOURS
from app.db.session import get_db
from app.models.models import User
from app.schemas.schemas import UserOut

# ── Crypto setup ─────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


# ── Password helpers ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    try:
        password_bytes = plain.encode('utf-8')
        hashed_bytes = hashed.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


# ── JWT helpers ──────────────────────────────────────────────────

def create_access_token(user: User) -> str:
    """Create a signed JWT with user identity and role claims."""
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "department": user.department,
        "full_name": user.full_name,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ─────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — validates Bearer token and returns the User ORM object.
    Use this in any route that requires authentication.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Malformed token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    return user


def require_role(*allowed_roles: str):
    """
    FastAPI dependency factory for role-based access control.

    Usage:
        @router.post("/upload")
        async def upload(current_user = Depends(require_role("admin", "manager"))):
            ...
    """
    async def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {list(allowed_roles)}. Your role: {current_user.role}",
            )
        return current_user
    return _check_role


# ── User CRUD ────────────────────────────────────────────────────

async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    full_name: str,
    role: str = "employee",
    department: str = "General",
) -> User:
    """Create a new user with hashed password."""
    # Check duplicate email
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        id=str(uuid.uuid4()),
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name,
        role=role,
        department=department,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    """Verify credentials. Returns User or raises 401."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    # Update last_login timestamp
    user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    return user
