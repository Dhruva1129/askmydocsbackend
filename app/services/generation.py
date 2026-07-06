"""
Generation Service
------------------
- Calls Groq (Llama 3 70B) — free tier, very fast
- Prompt forces the model to cite [chunk_id] inline
- Response parser validates every cited ID exists in context
- If citations are missing, one re-prompt attempt is made
- Streaming (SSE) support for token-by-token delivery
"""

import json
import re
import time
from typing import AsyncGenerator, List, Tuple

from groq import AsyncGroq

from app.core.config import settings
from app.schemas.schemas import ChunkResult

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY using the provided context chunks.
For every factual claim, cite the chunk ID inline using the format [chunk_id].
Example: "The model achieved 94% accuracy [abc123]."
If the context does not contain the answer, say: "I don't have enough information in the provided documents."
Do NOT make up information. Do NOT cite IDs that were not given to you."""


def _build_context_block(chunks: List[ChunkResult]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.chunk_id}] (source: {c.source_file}, page {c.page})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _build_messages(question: str, chunks: List[ChunkResult]) -> list:
    """Build the chat messages for the LLM call."""
    context_block = _build_context_block(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer (cite chunk IDs inline):"
            ),
        },
    ]


def _extract_cited_ids(answer: str) -> List[str]:
    """Pull all [chunk_id] references from the answer text."""
    return re.findall(r"\[([a-f0-9\-]{36})\]", answer)


def _validate_citations(cited_ids: List[str], valid_ids: set[str]) -> bool:
    """Returns True if ALL cited IDs are in the provided context."""
    return all(cid in valid_ids for cid in cited_ids)


async def _call_llm(messages: list) -> Tuple[str, float]:
    client = _get_client()
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices[0].message.content, round(latency_ms, 2)


# ═══════════════════════════════════════════════════════════════
#  Non-streaming generation (POST /query)
# ═══════════════════════════════════════════════════════════════

async def generate_answer(
    question: str,
    chunks: List[ChunkResult],
) -> Tuple[str, List[str], bool, float]:
    """
    Returns:
        answer (str)
        cited_chunk_ids (List[str])
        citation_valid (bool)
        llm_latency_ms (float)
    """
    valid_ids = {c.chunk_id for c in chunks}
    messages = _build_messages(question, chunks)

    answer, llm_latency_ms = await _call_llm(messages)
    cited_ids = _extract_cited_ids(answer)
    citation_valid = _validate_citations(cited_ids, valid_ids)

    # One re-prompt if citations are invalid
    if not citation_valid and cited_ids:
        bad_ids = [cid for cid in cited_ids if cid not in valid_ids]
        messages.append({"role": "assistant", "content": answer})
        messages.append({
            "role": "user",
            "content": (
                f"Your answer cited the following IDs that were NOT in the provided context: {bad_ids}. "
                "Please revise your answer using ONLY the chunk IDs listed above."
            ),
        })
        answer2, extra_latency = await _call_llm(messages)
        llm_latency_ms += extra_latency
        cited_ids = _extract_cited_ids(answer2)
        citation_valid = _validate_citations(cited_ids, valid_ids)
        answer = answer2

    return answer, cited_ids, citation_valid, llm_latency_ms


# ═══════════════════════════════════════════════════════════════
#  Streaming generation (SSE /query/stream)
# ═══════════════════════════════════════════════════════════════

async def generate_answer_stream(
    question: str,
    chunks: List[ChunkResult],
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted events for streaming responses.

    Event types:
      - event: chunk_context  → retrieval context (sent once at start)
      - event: token          → individual LLM token
      - event: citations      → extracted & validated citation data
      - event: done           → final metadata (latency, validation)
      - event: error          → error information
    """
    valid_ids = {c.chunk_id for c in chunks}
    messages = _build_messages(question, chunks)

    # ── Send context chunks to client ──────────────────────────
    context_data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_file": c.source_file,
            "page": c.page,
            "score": c.score,
        }
        for c in chunks
    ]
    yield _sse_event("chunk_context", {"chunks": context_data})

    # ── Stream LLM tokens ─────────────────────────────────────
    client = _get_client()
    full_answer = ""
    t0 = time.perf_counter()

    try:
        stream = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token = delta.content
                full_answer += token
                yield _sse_event("token", {"text": token})

    except Exception as e:
        yield _sse_event("error", {"message": str(e)})
        return

    llm_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── Citation extraction & validation ──────────────────────
    cited_ids = _extract_cited_ids(full_answer)
    citation_valid = _validate_citations(cited_ids, valid_ids)

    # One re-prompt attempt if citations are invalid (non-streamed)
    if not citation_valid and cited_ids:
        bad_ids = [cid for cid in cited_ids if cid not in valid_ids]
        messages.append({"role": "assistant", "content": full_answer})
        messages.append({
            "role": "user",
            "content": (
                f"Your answer cited the following IDs that were NOT in the provided context: {bad_ids}. "
                "Please revise your answer using ONLY the chunk IDs listed above."
            ),
        })

        yield _sse_event("reprompt", {"reason": f"Invalid citations: {bad_ids}"})

        # Re-prompt is sent non-streamed for simplicity, then re-stream the corrected answer
        try:
            full_answer = ""
            t1 = time.perf_counter()
            retry_stream = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in retry_stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token = delta.content
                    full_answer += token
                    yield _sse_event("token", {"text": token})

            llm_ms += round((time.perf_counter() - t1) * 1000, 2)
            cited_ids = _extract_cited_ids(full_answer)
            citation_valid = _validate_citations(cited_ids, valid_ids)
        except Exception as e:
            yield _sse_event("error", {"message": f"Re-prompt failed: {e}"})

    # ── Final metadata ────────────────────────────────────────
    cited_chunks = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_file": c.source_file,
            "page": c.page,
            "score": c.score,
        }
        for c in chunks if c.chunk_id in set(cited_ids)
    ]

    yield _sse_event("citations", {
        "cited_ids": cited_ids,
        "cited_chunks": cited_chunks,
        "citation_valid": citation_valid,
    })

    yield _sse_event("done", {
        "answer": full_answer,
        "llm_ms": llm_ms,
        "citation_valid": citation_valid,
    })


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return json.dumps({"event": event, **data})
