"""
Security Service
-----------------
Covers all security layers between the user and the LLM:
  1. build_permission_filter()   → ChromaDB where clause from user role
  2. detect_prompt_injection()   → block malicious inputs
  3. scan_chunks_for_sensitive() → mask/flag PII in retrieved chunks
  4. validate_output()           → scan LLM response for leaked data
  5. mask_sensitive_patterns()   → regex-based masking utility
  6. write_audit_log()           → persist full audit trail
"""

import hashlib
import json
from typing import List, Optional, Tuple, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    INJECTION_PATTERNS,
    SENSITIVE_PATTERNS,
    ROLE_DEPARTMENT_ACCESS,
    ROLE_ACCESS_LEVEL,
    SALARY_ALLOWED_ROLES,
)
from app.models.models import AuditLog, User
from app.schemas.schemas import ChunkResult
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
#  1. Permission Filter Builder
# ═══════════════════════════════════════════════════════════════

def build_permission_filter(user: User, domain: Optional[str] = None) -> Optional[dict]:
    """
    Build a ChromaDB `where` filter based on the user's role.

    ChromaDB supports $in, $eq, $and, $or operators.
    We filter on the `department` and `access_level` stored in chunk metadata.

    Note: ChromaDB where clauses are limited — we use $in for access_level
    and do a post-filter for allowed_roles (stored as JSON string in metadata).
    """
    allowed_levels = ROLE_ACCESS_LEVEL.get(user.role, ["public"])
    allowed_depts = list(ROLE_DEPARTMENT_ACCESS.get(user.role, {"General"}))

    filters = []

    # Filter by access level
    if len(allowed_levels) == 1:
        filters.append({"access_level": {"$eq": allowed_levels[0]}})
    else:
        filters.append({"access_level": {"$in": allowed_levels}})

    # Filter by department (users only get their authorized departments)
    if len(allowed_depts) == 1:
        filters.append({"department": {"$eq": allowed_depts[0]}})
    else:
        filters.append({"department": {"$in": allowed_depts}})

    # Domain filter if specified
    if domain:
        filters.append({"domain": {"$eq": domain}})

    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def post_filter_chunks_by_role(chunks_raw: list, user: User) -> list:
    """
    Secondary filter after ChromaDB retrieval.
    Checks the `allowed_roles` JSON field stored in each chunk's metadata.
    Falls back to keeping the chunk if no allowed_roles set (backwards compat).
    """
    filtered = []
    for item in chunks_raw:
        meta = item.get("meta", {})
        allowed_roles_raw = meta.get("allowed_roles")

        if not allowed_roles_raw:
            # No restriction set → accessible to all (public docs)
            filtered.append(item)
            continue

        # Parse JSON string stored in ChromaDB
        try:
            allowed_roles = json.loads(allowed_roles_raw) if isinstance(allowed_roles_raw, str) else allowed_roles_raw
        except (json.JSONDecodeError, TypeError):
            allowed_roles = []

        if not allowed_roles or user.role in allowed_roles or user.role == "admin":
            filtered.append(item)

    return filtered


# ═══════════════════════════════════════════════════════════════
#  2. Prompt Injection Detection
# ═══════════════════════════════════════════════════════════════

def detect_prompt_injection(query: str) -> Tuple[bool, Optional[str]]:
    """
    Scans the query for prompt injection patterns.

    Returns:
        (is_injection, matched_pattern_description)
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            return True, f"Detected pattern: '{match.group(0)}'"
    return False, None


# ═══════════════════════════════════════════════════════════════
#  3. Sensitive Data Scanner (Pre-LLM)
# ═══════════════════════════════════════════════════════════════

def scan_chunks_for_sensitive(
    chunks: List[ChunkResult],
    user: User,
) -> Tuple[List[ChunkResult], bool, List[str]]:
    """
    Scans retrieved chunks BEFORE sending to LLM.
    - Masks sensitive values if user role doesn't allow them
    - Returns (filtered_chunks, any_masked, sensitive_types_found)
    """
    any_masked = False
    sensitive_types_found = []
    filtered_chunks = []

    for chunk in chunks:
        text = chunk.text
        masked_text = text
        chunk_masked = False

        for pattern_name, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                # Role-based decision
                if pattern_name == "salary" and user.role in SALARY_ALLOWED_ROLES:
                    # User is allowed to see salary data — keep it
                    continue

                # Mask the sensitive data
                masked_text = pattern.sub(f"[{pattern_name.upper()} REDACTED]", masked_text)
                if pattern_name not in sensitive_types_found:
                    sensitive_types_found.append(pattern_name)
                chunk_masked = True
                any_masked = True

        if chunk_masked:
            # Return a new ChunkResult with masked text
            filtered_chunks.append(chunk.model_copy(update={"text": masked_text}))
        else:
            filtered_chunks.append(chunk)

    return filtered_chunks, any_masked, sensitive_types_found


# ═══════════════════════════════════════════════════════════════
#  4. Output Validator (Post-LLM)
# ═══════════════════════════════════════════════════════════════

def validate_output(response: str, user: User) -> Tuple[str, bool, List[str]]:
    """
    Scans the LLM response for accidentally leaked sensitive data.
    Returns (cleaned_response, was_masked, types_masked)
    """
    cleaned = response
    was_masked = False
    types_masked = []

    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(cleaned):
            if pattern_name == "salary" and user.role in SALARY_ALLOWED_ROLES:
                continue
            cleaned = pattern.sub(f"[{pattern_name.upper()} REDACTED]", cleaned)
            types_masked.append(pattern_name)
            was_masked = True

    return cleaned, was_masked, types_masked


# ═══════════════════════════════════════════════════════════════
#  5. Audit Logger
# ═══════════════════════════════════════════════════════════════

async def write_audit_log(
    db: AsyncSession,
    user: User,
    query: str,
    retrieved_chunks: List[ChunkResult],
    injection_detected: bool,
    injection_pattern: Optional[str],
    sensitive_blocked: bool,
    sensitive_types: List[str],
    response_status: str,  # "success" | "blocked_injection" | "blocked_permission" | "error"
    latency_ms: float,
) -> None:
    """
    Writes a full audit entry to the audit_logs table.
    Call this on every query — regardless of success or failure.
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    log = AuditLog(
        user_id=user.id,
        user_email=user.email,
        user_role=user.role,
        query=query,
        query_hash=query_hash,
        retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
        retrieved_doc_names=list({c.source_file for c in retrieved_chunks}),
        injection_detected=injection_detected,
        injection_pattern=injection_pattern,
        sensitive_blocked=sensitive_blocked,
        sensitive_types=sensitive_types,
        response_status=response_status,
        latency_ms=round(latency_ms, 2),
    )
    db.add(log)
    await db.commit()
