from sqlalchemy import String, Integer, Float, Text, DateTime, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone

from app.db.session import Base


class User(Base):
    """Authenticated user with RBAC role."""
    __tablename__ = "users"

    id: Mapped[str]            = mapped_column(String(36), primary_key=True)   # UUID
    email: Mapped[str]         = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str]     = mapped_column(String(255), default="")
    role: Mapped[str]          = mapped_column(String(50), default="employee")
    # admin | hr | finance | manager | developer | employee
    department: Mapped[str]    = mapped_column(String(100), default="General")
    is_active: Mapped[bool]    = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Document(Base):
    """Ingested document with security metadata."""
    __tablename__ = "documents"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str]    = mapped_column(String(255), nullable=False)
    domain: Mapped[str]      = mapped_column(String(100), default="general")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str]      = mapped_column(String(50), default="processing")
    # processing | ready | error

    # ── Security metadata (NEW) ──────────────────────────────────
    department: Mapped[str]           = mapped_column(String(100), default="General")
    access_level: Mapped[str]         = mapped_column(String(50), default="internal")
    # public | internal | confidential | secret
    allowed_roles: Mapped[list]       = mapped_column(JSON, default=list)
    # e.g. ["admin", "hr", "manager"]
    document_type: Mapped[str]        = mapped_column(String(50), default="general")
    # policy | financial | hr_data | technical | general | contract
    confidentiality_level: Mapped[str] = mapped_column(String(50), default="low")
    # low | medium | high | critical
    owner_email: Mapped[str]          = mapped_column(String(255), default="")
    uploaded_by: Mapped[str]          = mapped_column(String(36), default="")
    # FK → users.id (not enforced at DB level for flexibility)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class QueryLog(Base):
    """Logs every RAG query for analytics."""
    __tablename__ = "query_logs"

    id: Mapped[int]                     = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str]               = mapped_column(Text, nullable=False)
    answer: Mapped[str]                 = mapped_column(Text, nullable=False)
    cited_chunk_ids: Mapped[list]       = mapped_column(JSON, default=list)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0)
    rerank_latency_ms: Mapped[float]    = mapped_column(Float, default=0)
    llm_latency_ms: Mapped[float]       = mapped_column(Float, default=0)
    total_latency_ms: Mapped[float]     = mapped_column(Float, default=0)
    created_at: Mapped[datetime]        = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class AuditLog(Base):
    """Full audit trail for every query attempt — compliance & traceability."""
    __tablename__ = "audit_logs"

    id: Mapped[int]                = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str]           = mapped_column(String(36), nullable=False)
    user_email: Mapped[str]        = mapped_column(String(255), nullable=False)
    user_role: Mapped[str]         = mapped_column(String(50), nullable=False)
    query: Mapped[str]             = mapped_column(Text, nullable=False)
    query_hash: Mapped[str]        = mapped_column(String(64), nullable=False)  # SHA-256
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_doc_names: Mapped[list] = mapped_column(JSON, default=list)
    injection_detected: Mapped[bool]  = mapped_column(Boolean, default=False)
    injection_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    sensitive_blocked: Mapped[bool]   = mapped_column(Boolean, default=False)
    sensitive_types: Mapped[list]     = mapped_column(JSON, default=list)
    response_status: Mapped[str]      = mapped_column(String(50), default="success")
    # success | blocked_injection | blocked_permission | error
    latency_ms: Mapped[float]         = mapped_column(Float, default=0)
    created_at: Mapped[datetime]      = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class EvalRun(Base):
    """Stores every RAGAS evaluation run for quality tracking."""
    __tablename__ = "eval_runs"

    id: Mapped[int]                 = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_sha: Mapped[str]         = mapped_column(String(40), default="local")
    faithfulness: Mapped[float]     = mapped_column(Float)
    answer_relevancy: Mapped[float] = mapped_column(Float)
    context_recall: Mapped[float]   = mapped_column(Float)
    passed: Mapped[bool]            = mapped_column(Boolean, default=False)
    details: Mapped[dict]           = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict]   = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]    = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
