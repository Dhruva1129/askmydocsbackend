from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime


# ── Auth ────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds
    user: "UserOut"


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8, description="Min 8 chars")
    full_name: str = Field(..., min_length=2)
    role: str = Field(default="employee")
    department: str = Field(default="General")


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    department: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Documents ──────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: int
    filename: str
    domain: str
    chunk_count: int
    status: str
    department: str
    access_level: str
    allowed_roles: List[str]
    document_type: str
    confidentiality_level: str
    owner_email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Retrieval ───────────────────────────────────────────────────

class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page: Optional[int] = None
    score: float
    department: Optional[str] = None
    access_level: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    chunks: List[ChunkResult]
    retrieval_ms: float
    rerank_ms: float


# ── Query ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    domain: Optional[str] = None
    top_k: Optional[int] = Field(5, ge=1, le=20)


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    llm_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    cited_chunks: List[ChunkResult]
    latency: LatencyBreakdown
    citation_valid: bool
    sensitive_masked: bool = False     # True if any PII was masked in response
    injection_detected: bool = False   # Always False here (blocked before LLM)


# ── Audit Logs ─────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id: int
    user_id: str
    user_email: str
    user_role: str
    query: str
    retrieved_doc_names: List[str]
    injection_detected: bool
    sensitive_blocked: bool
    response_status: str
    latency_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Eval ────────────────────────────────────────────────────────

class EvalRunOut(BaseModel):
    id: int
    commit_sha: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    passed: bool
    config_snapshot: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# Required for TokenResponse forward reference
TokenResponse.model_rebuild()
