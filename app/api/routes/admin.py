"""
Admin Routes
------------
All routes require the 'admin' role.

GET  /admin/users                   List all users
PUT  /admin/users/{user_id}/role    Change user role
DELETE /admin/users/{user_id}       Deactivate user
GET  /admin/audit-logs              Query audit logs with filters
GET  /admin/audit-logs/export       Export audit logs as CSV text
"""

import csv
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.models.models import User, AuditLog
from app.schemas.schemas import UserOut, AuditLogOut
from app.services.auth import get_current_user, require_role
from app.core.security import UserRole

router = APIRouter()

admin_only = require_role("admin")


# ── User Management ──────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
):
    """List all registered users."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


@router.put("/users/{user_id}/role")
async def change_user_role(
    user_id: str,
    new_role: str,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(admin_only),
):
    """Change a user's role. Admin only."""
    valid_roles = [r.value for r in UserRole]
    if new_role not in valid_roles:
        raise HTTPException(400, f"Invalid role. Must be one of: {valid_roles}")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == current_admin.id:
        raise HTTPException(400, "Cannot change your own role")

    old_role = user.role
    user.role = new_role
    await db.commit()
    return {"message": f"Role updated from '{old_role}' to '{new_role}'", "user_id": user_id}


@router.delete("/users/{user_id}")
async def deactivate_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(admin_only),
):
    """Deactivate (soft-delete) a user account."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == current_admin.id:
        raise HTTPException(400, "Cannot deactivate your own account")

    user.is_active = False
    await db.commit()
    return {"message": f"User {user.email} deactivated"}


# ── Audit Logs ────────────────────────────────────────────────────

@router.get("/audit-logs", response_model=list[AuditLogOut])
async def get_audit_logs(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
    user_email: Optional[str] = Query(None, description="Filter by user email"),
    role: Optional[str] = Query(None, description="Filter by user role"),
    injection_only: bool = Query(False, description="Only show injection attempts"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Query audit logs with optional filters. Returns most recent first."""
    query = select(AuditLog).order_by(desc(AuditLog.created_at))

    if user_email:
        query = query.where(AuditLog.user_email == user_email)
    if role:
        query = query.where(AuditLog.user_role == role)
    if injection_only:
        query = query.where(AuditLog.injection_detected == True)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/audit-logs/export")
async def export_audit_logs(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Export audit logs as a downloadable CSV file."""
    result = await db.execute(
        select(AuditLog).order_by(desc(AuditLog.created_at)).limit(limit)
    )
    logs = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp", "user_email", "user_role",
        "query", "response_status", "injection_detected",
        "sensitive_blocked", "sensitive_types", "latency_ms",
    ])
    for log in logs:
        writer.writerow([
            log.id,
            log.created_at.isoformat(),
            log.user_email,
            log.user_role,
            log.query[:200],   # truncate long queries in CSV
            log.response_status,
            log.injection_detected,
            log.sensitive_blocked,
            ",".join(log.sensitive_types or []),
            log.latency_ms,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
    )
