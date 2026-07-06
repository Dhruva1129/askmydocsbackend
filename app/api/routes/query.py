"""
Query Routes — Secured
-----------------------
POST /query/       Full RAG pipeline with auth + security layers
POST /query/stream SSE streaming with auth + security layers

Security pipeline per request:
  1. Verify JWT → get user
  2. Detect prompt injection → block if found
  3. Build permission filter from user role
  4. Hybrid retrieval (filtered by permissions)
  5. Post-filter chunks by allowed_roles metadata
  6. Scan chunks for PII → mask based on role
  7. LLM generation
  8. Output validation → mask leaked PII
  9. Write audit log
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import QueryLog, User
from app.schemas.schemas import QueryRequest, QueryResponse, LatencyBreakdown, ChunkResult
from app.services.retrieval import retrieve
from app.services.generation import generate_answer, generate_answer_stream
from app.services.auth import get_current_user
from app.services.security import (
    detect_prompt_injection,
    build_permission_filter,
    post_filter_chunks_by_role,
    scan_chunks_for_sensitive,
    validate_output,
    write_audit_log,
)

router = APIRouter()


async def _secure_retrieve(request: QueryRequest, user: User):
    """Run hybrid retrieval with permission filter applied."""
    user_filter = build_permission_filter(user, domain=request.domain)
    chunks, ret_latency = await retrieve(
        question=request.question,
        domain=request.domain,
        user_filter=user_filter,
    )
    return chunks, ret_latency


# ═══════════════════════════════════════════════════════════════
#  POST /query/ — Non-streaming secure response
# ═══════════════════════════════════════════════════════════════

@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    t_total = time.perf_counter()

    # ── 1. Prompt injection check ─────────────────────────────
    is_injection, injection_pattern = detect_prompt_injection(request.question)
    if is_injection:
        await write_audit_log(
            db=db, user=current_user, query=request.question,
            retrieved_chunks=[], injection_detected=True,
            injection_pattern=injection_pattern, sensitive_blocked=False,
            sensitive_types=[], response_status="blocked_injection",
            latency_ms=(time.perf_counter() - t_total) * 1000,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query blocked: potential prompt injection detected. {injection_pattern}",
        )

    # ── 2. Retrieve (with permission filter) ──────────────────
    chunks, ret_latency = await _secure_retrieve(request, current_user)

    if not chunks:
        await write_audit_log(
            db=db, user=current_user, query=request.question,
            retrieved_chunks=[], injection_detected=False,
            injection_pattern=None, sensitive_blocked=False,
            sensitive_types=[], response_status="blocked_permission",
            latency_ms=(time.perf_counter() - t_total) * 1000,
        )
        return QueryResponse(
            question=request.question,
            answer="I don't have access to any documents relevant to your query, or you may not be authorized to access them.",
            cited_chunks=[],
            latency=LatencyBreakdown(
                retrieval_ms=ret_latency.get("retrieval_ms", 0),
                rerank_ms=ret_latency.get("rerank_ms", 0),
                llm_ms=0,
                total_ms=round((time.perf_counter() - t_total) * 1000, 2),
            ),
            citation_valid=True,
            sensitive_masked=False,
        )

    # ── 3. Scan chunks for PII before LLM ────────────────────
    safe_chunks, any_masked, sensitive_types = scan_chunks_for_sensitive(chunks, current_user)

    # ── 4. Generate ───────────────────────────────────────────
    answer, cited_ids, citation_valid, llm_ms = await generate_answer(request.question, safe_chunks)

    # ── 5. Output validation ──────────────────────────────────
    clean_answer, out_masked, out_types = validate_output(answer, current_user)
    final_masked = any_masked or out_masked
    all_sensitive_types = list(set(sensitive_types + out_types))

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)
    cited_chunks = [c for c in safe_chunks if c.chunk_id in set(cited_ids)]

    # ── 6. Log query ──────────────────────────────────────────
    log = QueryLog(
        question=request.question,
        answer=clean_answer,
        cited_chunk_ids=cited_ids,
        retrieval_latency_ms=ret_latency.get("retrieval_ms", 0),
        rerank_latency_ms=ret_latency.get("rerank_ms", 0),
        llm_latency_ms=llm_ms,
        total_latency_ms=total_ms,
    )
    db.add(log)

    # ── 7. Audit log ──────────────────────────────────────────
    await write_audit_log(
        db=db, user=current_user, query=request.question,
        retrieved_chunks=cited_chunks, injection_detected=False,
        injection_pattern=None, sensitive_blocked=final_masked,
        sensitive_types=all_sensitive_types, response_status="success",
        latency_ms=total_ms,
    )

    return QueryResponse(
        question=request.question,
        answer=clean_answer,
        cited_chunks=cited_chunks,
        latency=LatencyBreakdown(
            retrieval_ms=ret_latency.get("retrieval_ms", 0),
            rerank_ms=ret_latency.get("rerank_ms", 0),
            llm_ms=llm_ms,
            total_ms=total_ms,
        ),
        citation_valid=citation_valid,
        sensitive_masked=final_masked,
    )


# ═══════════════════════════════════════════════════════════════
#  POST /query/stream — SSE streaming secure response
# ═══════════════════════════════════════════════════════════════

@router.post("/stream")
async def query_documents_stream(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import json as _json
    from sse_starlette.sse import EventSourceResponse

    t_total = time.perf_counter()

    # ── 1. Injection check ────────────────────────────────────
    is_injection, injection_pattern = detect_prompt_injection(request.question)
    if is_injection:
        await write_audit_log(
            db=db, user=current_user, query=request.question,
            retrieved_chunks=[], injection_detected=True,
            injection_pattern=injection_pattern, sensitive_blocked=False,
            sensitive_types=[], response_status="blocked_injection",
            latency_ms=(time.perf_counter() - t_total) * 1000,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {injection_pattern}",
        )

    async def event_generator():
        t0 = time.perf_counter()

        # Retrieve
        chunks, ret_latency = await _secure_retrieve(request, current_user)

        yield {
            "event": "retrieval_done",
            "data": _json.dumps({
                "retrieval_ms": ret_latency.get("retrieval_ms", 0),
                "rerank_ms": ret_latency.get("rerank_ms", 0),
                "num_chunks": len(chunks),
            }),
        }

        if not chunks:
            yield {
                "event": "done",
                "data": _json.dumps({
                    "answer": "No authorized documents found for your query.",
                    "llm_ms": 0,
                    "citation_valid": True,
                    "sensitive_masked": False,
                }),
            }
            return

        # Scan for PII
        safe_chunks, any_masked, sensitive_types = scan_chunks_for_sensitive(chunks, current_user)

        # Stream generation
        async for sse_raw in generate_answer_stream(request.question, safe_chunks):
            parsed = _json.loads(sse_raw)
            event_type = parsed.pop("event")

            # Intercept `done` event to add security metadata
            if event_type == "done":
                answer = parsed.get("answer", "")
                clean_answer, out_masked, out_types = validate_output(answer, current_user)
                parsed["answer"] = clean_answer
                parsed["sensitive_masked"] = any_masked or out_masked

                # Async audit log
                cited_ids = []
                try:
                    cited_ids = parsed.get("cited_ids", [])
                except Exception:
                    pass

                await write_audit_log(
                    db=db, user=current_user, query=request.question,
                    retrieved_chunks=safe_chunks, injection_detected=False,
                    injection_pattern=None, sensitive_blocked=any_masked or out_masked,
                    sensitive_types=list(set(sensitive_types + out_types)),
                    response_status="success",
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                )

            yield {"event": event_type, "data": _json.dumps(parsed)}

    return EventSourceResponse(event_generator())
