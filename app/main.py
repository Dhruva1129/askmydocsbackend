from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import documents, query, eval, retrieve, auth, admin
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AskMyDocs — Secure Enterprise RAG API",
    description=(
        "Production RAG system with hybrid retrieval, reranking, "
        "RBAC authentication, prompt injection protection, PII masking, "
        "output validation, and full audit logging."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Public routes (no auth required) ─────────────────────────────
app.include_router(auth.router,      prefix="/auth",         tags=["Authentication"])

# ── Secured routes ────────────────────────────────────────────────
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])
app.include_router(eval.router,      prefix="/api/eval",      tags=["Evaluation"])
app.include_router(retrieve.router,  prefix="/api",           tags=["Retrieval"])

# ── Admin routes (admin role only) ────────────────────────────────
app.include_router(admin.router,     prefix="/admin",         tags=["Admin"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "2.0.0", "auth": "enabled"}
