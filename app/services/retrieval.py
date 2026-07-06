"""
Retrieval Service
-----------------
Step 1: Vector search        — top-20 from ChromaDB (with permission filter)
Step 2: BM25 keyword search  — top-20 from in-memory BM25Okapi index
         → Post-filtered by user permissions via ChromaDB metadata lookup
Step 3: Reciprocal Rank Fusion (RRF) — merge & re-rank both lists
Step 4: Cross-encoder reranking — score top-10 RRF candidates
Step 5: Return top-5 chunks to the generation stage
"""

import json
import time
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.schemas.schemas import ChunkResult
from app.services.ingestion import get_collection, get_embedder, get_bm25

# Cross-encoder loaded once (small, free model)
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    RRF formula: score(d) = Σ 1 / (k + rank(d))
    Returns {chunk_id: rrf_score} sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def _chunk_passes_filter(meta: dict, user_filter: Optional[dict]) -> bool:
    """
    Post-retrieval check: does this chunk's metadata satisfy the user_filter?
    Used to filter BM25 results that bypass ChromaDB's where clause.

    Handles $and, $in, $eq operators matching ChromaDB filter syntax.
    Falls back to True (allow) if the filter is None or metadata is missing fields.
    """
    if user_filter is None:
        return True

    def _check(fltr: dict, m: dict) -> bool:
        # $and combinator
        if "$and" in fltr:
            return all(_check(sub, m) for sub in fltr["$and"])
        # $or combinator
        if "$or" in fltr:
            return any(_check(sub, m) for sub in fltr["$or"])

        # Field-level check
        for field, condition in fltr.items():
            if field.startswith("$"):
                continue
            value = m.get(field)
            if value is None:
                # Field missing from metadata → old pre-security chunk
                # Allow it through so existing indexed docs still work
                return True
            if isinstance(condition, dict):
                op = list(condition.keys())[0]
                expected = condition[op]
                if op == "$eq":
                    if value != expected:
                        return False
                elif op == "$in":
                    if value not in expected:
                        return False
                elif op == "$ne":
                    if value == expected:
                        return False
            else:
                if value != condition:
                    return False
        return True

    return _check(user_filter, meta)


async def retrieve(
    question: str,
    domain: str | None = None,
    user_filter: dict | None = None,
) -> Tuple[List[ChunkResult], dict]:
    """
    Returns (ranked_chunks, latency_dict).
    latency_dict keys: retrieval_ms, rerank_ms
    user_filter: optional ChromaDB where clause from security service
    """
    t0 = time.perf_counter()

    # ── 1. Vector search (with permission filter) ─────────────
    embedder = get_embedder()
    query_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()

    collection = get_collection()
    where = user_filter  # starts with permission filter
    if domain:
        domain_filter = {"domain": {"$eq": domain}}
        if where:
            where = {"$and": [where, domain_filter]}
        else:
            where = domain_filter

    # Guard: if filter matches 0 docs ChromaDB raises; catch and treat as empty
    try:
        vec_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.VECTOR_TOP_K,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids: List[str] = vec_results["ids"][0]
        vec_docs: List[str] = vec_results["documents"][0]
        vec_metas: List[dict] = vec_results["metadatas"][0]
    except Exception:
        # Filter matched zero documents — treat as empty vector results
        vec_ids, vec_docs, vec_metas = [], [], []

    # Build a lookup: chunk_id → {text, meta}
    chunk_lookup: dict[str, dict] = {}
    for cid, doc, meta in zip(vec_ids, vec_docs, vec_metas):
        # Double-check: only keep chunks that pass permission filter
        if _chunk_passes_filter(meta, user_filter):
            chunk_lookup[cid] = {"text": doc, "meta": meta}

    # Rebuild vec_ids to only include filtered ones
    vec_ids = [cid for cid in vec_ids if cid in chunk_lookup]

    # ── 2. BM25 keyword search (+ permission post-filter) ─────
    bm25_index, bm25_corpus = get_bm25()
    bm25_ids: List[str] = []

    if bm25_index and bm25_corpus:
        tokenized_query = question.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][: settings.BM25_TOP_K]

        # Collect BM25 candidates that have a positive score
        bm25_candidates = [
            (bm25_corpus[idx]["chunk_id"], bm25_scores[idx])
            for idx in top_bm25_idx
            if bm25_scores[idx] > 0
        ]

        # Fetch metadata for BM25 chunks NOT already in chunk_lookup
        new_bm25_ids = [cid for cid, _ in bm25_candidates if cid not in chunk_lookup]
        if new_bm25_ids:
            try:
                fetched = collection.get(
                    ids=new_bm25_ids,
                    include=["documents", "metadatas"],
                )
                for cid, doc, meta in zip(
                    fetched["ids"], fetched["documents"], fetched["metadatas"]
                ):
                    # Apply permission filter to BM25 results too
                    if _chunk_passes_filter(meta, user_filter):
                        chunk_lookup[cid] = {"text": doc, "meta": meta}
            except Exception:
                pass

        # Only include BM25 results that survived the permission filter
        for cid, _ in bm25_candidates:
            if cid in chunk_lookup:
                bm25_ids.append(cid)

    # ── 3. RRF fusion ─────────────────────────────────────────
    rrf_scores = _reciprocal_rank_fusion([vec_ids, bm25_ids])
    fused_ids = list(rrf_scores.keys())

    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Early exit: nothing retrieved at all
    if not fused_ids:
        return [], {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": 0.0,
        }

    # ── 4. Cross-encoder reranking (top-10 RRF candidates) ────
    t1 = time.perf_counter()

    PRE_RERANK_TOP_K = 10
    candidates = [
        (cid, chunk_lookup[cid]["text"])
        for cid in fused_ids[:PRE_RERANK_TOP_K]
        if cid in chunk_lookup
    ]

    if not candidates:
        return [], {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": 0.0,
        }

    cross_encoder = _get_cross_encoder()
    pairs = [[question, text] for _, text in candidates]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, ce_scores),
        key=lambda x: x[1],
        reverse=True,
    )[: settings.RERANK_TOP_K]

    rerank_ms = (time.perf_counter() - t1) * 1000

    # ── Build response ────────────────────────────────────────
    results = []
    for (cid, text), score in reranked:
        meta = chunk_lookup[cid]["meta"]
        results.append(ChunkResult(
            chunk_id=cid,
            text=text,
            source_file=meta.get("source_file", "unknown"),
            page=meta.get("source_page"),
            score=float(score),
            department=meta.get("department"),
            access_level=meta.get("access_level"),
        ))

    return results, {
        "retrieval_ms": round(retrieval_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
    }
