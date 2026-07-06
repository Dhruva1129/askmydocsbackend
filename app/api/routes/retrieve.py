from fastapi import APIRouter, Query, Depends
from typing import Optional

from app.schemas.schemas import RetrieveResponse
from app.services.retrieval import retrieve
from app.services.auth import get_current_user
from app.services.security import build_permission_filter
from app.models.models import User

router = APIRouter()


@router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(
    q: str = Query(..., min_length=3, max_length=1000, description="Search query"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Hybrid retrieval endpoint — returns top-ranked chunks for a query.
    Results are filtered by the current user's role and permissions.
    Requires authentication.
    """
    user_filter = build_permission_filter(current_user, domain=domain)
    chunks, latency = await retrieve(
        question=q,
        domain=domain,
        user_filter=user_filter,
    )
    return RetrieveResponse(
        query=q,
        chunks=chunks[:top_k],
        retrieval_ms=latency.get("retrieval_ms", 0),
        rerank_ms=latency.get("rerank_ms", 0),
    )
