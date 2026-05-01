"""
hybrid_retriever.py – Hybrid keyword + semantic retrieval.

Production Improvements:
- Score normalization
- Soft-threshold fallback support
- Reduced retrieval size
- Safe ID handling
- Strong semantic availability detection
- No duplicate score computation
- Delayed rounding for accuracy
- top_k safety cap
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.config import settings
from src.data.knowledge_base import KnowledgeChunk, get_all_chunks
from src.retrieval.embedding_retriever import semantic_retrieve
from src.retrieval.keyword_retriever import keyword_retrieve

logger = logging.getLogger(__name__)


def _normalize_scores(
    results: List[Tuple[KnowledgeChunk, float]],
) -> Dict[str, float]:
    """
    Min-max normalize scores into [0, 1].
    """

    if not results:
        return {}

    raw_scores = [score for _, score in results]

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    # Avoid divide-by-zero
    if max_score == min_score:
        return {
            chunk.id: 1.0
            for chunk, _ in results
            if getattr(chunk, "id", None)
        }

    normalized = {}

    for chunk, score in results:

        chunk_id = getattr(chunk, "id", None)

        if not chunk_id:
            continue

        normalized[chunk_id] = (
            (score - min_score)
            / (max_score - min_score)
        )

    return normalized


def _compute_hybrid_scores(
    all_ids: set[str],
    chunk_lookup: Dict[str, KnowledgeChunk],
    kw_map: Dict[str, float],
    sem_map: Dict[str, float],
    semantic_available: bool,
) -> List[Tuple[KnowledgeChunk, float]]:
    """
    Compute hybrid scores once.
    Prevents duplicate computation during fallback logic.
    """

    results: List[Tuple[KnowledgeChunk, float]] = []

    for chunk_id in all_ids:

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            logger.warning(
                f"[HybridRetriever] Missing chunk for ID={chunk_id}"
            )
            continue

        kw_score = kw_map.get(chunk_id, 0.0)
        sem_score = sem_map.get(chunk_id, 0.0)

        if semantic_available:
            hybrid_score = (
                settings.KEYWORD_WEIGHT * kw_score
                + settings.SEMANTIC_WEIGHT * sem_score
            )
        else:
            hybrid_score = kw_score

        results.append((chunk, hybrid_score))

    return results


def hybrid_retrieve(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> List[Tuple[KnowledgeChunk, float]]:
    """
    Hybrid retrieval pipeline.
    """

    # ─────────────────────────────────────────────────────────────
    # Safe top_k handling
    # ─────────────────────────────────────────────────────────────
    top_k = top_k or settings.TOP_K_CHUNKS

    if top_k <= 0:
        logger.warning(
            f"[HybridRetriever] Invalid top_k={top_k}. Using default."
        )
        top_k = settings.TOP_K_CHUNKS

    top_k = min(top_k, settings.MAX_RETRIEVAL_LIMIT)

    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.SIMILARITY_THRESHOLD
    )

    retrieval_limit = max(
        top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER,
        10,
    )

    chunks = get_all_chunks()

    if not chunks:
        logger.warning("[HybridRetriever] Knowledge base is empty.")
        return []

    # ─────────────────────────────────────────────────────────────
    # Run retrievers
    # ─────────────────────────────────────────────────────────────
    kw_results = keyword_retrieve(
        query=query,
        chunks=chunks,
        top_k=retrieval_limit,
    )

    sem_results = semantic_retrieve(
        query=query,
        chunks=chunks,
        top_k=retrieval_limit,
    )

    # ─────────────────────────────────────────────────────────────
    # Semantic gate: check raw scores BEFORE normalization
    # Min-max normalization inflates weak scores — an off-topic
    # query with raw scores [0.21, 0.19, 0.18] becomes [1.0, 0.5, 0.33]
    # after normalization, sailing past any post-fusion threshold.
    # Gate on the raw top score to reject genuinely irrelevant queries.
    # ─────────────────────────────────────────────────────────────
    MIN_SEMANTIC_SCORE = 0.45  # tune based on your embedding model

    if sem_results and sem_results[0][1] < MIN_SEMANTIC_SCORE:
        logger.info(
            f"[HybridRetriever] Semantic gate failed — "
            f"top raw score={sem_results[0][1]:.4f} < {MIN_SEMANTIC_SCORE}. "
            f"Query likely off-topic. Returning empty."
        )
        return []

    # Stronger semantic availability detection
    semantic_available = (
        bool(sem_results)
        and any(score > 0 for _, score in sem_results)
    )

    if not semantic_available:
        logger.warning(
            "[HybridRetriever] Semantic retrieval unavailable. "
            "Using keyword-only retrieval."
        )

    # ─────────────────────────────────────────────────────────────
    # Normalize scores
    # ─────────────────────────────────────────────────────────────
    kw_map = _normalize_scores(kw_results)
    sem_map = _normalize_scores(sem_results)

    # Safe lookup
    chunk_lookup: Dict[str, KnowledgeChunk] = {
        c.id: c
        for c in chunks
        if getattr(c, "id", None)
    }

    all_ids = set(kw_map.keys()) | set(sem_map.keys())

    # ─────────────────────────────────────────────────────────────
    # Compute hybrid scores ONCE
    # ─────────────────────────────────────────────────────────────
    all_results = _compute_hybrid_scores(
        all_ids=all_ids,
        chunk_lookup=chunk_lookup,
        kw_map=kw_map,
        sem_map=sem_map,
        semantic_available=semantic_available,
    )

    # ─────────────────────────────────────────────────────────────
    # Main threshold filtering
    # ─────────────────────────────────────────────────────────────
    filtered = [
        (chunk, score)
        for chunk, score in all_results
        if score >= threshold
    ]

    # ─────────────────────────────────────────────────────────────
    # Soft fallback
    # ─────────────────────────────────────────────────────────────
    if not filtered:

        soft_threshold = (
            threshold * settings.SOFT_THRESHOLD_FACTOR
        )

        logger.info(
            f"[HybridRetriever] "
            f"No results above threshold={threshold:.4f}. "
            f"Trying soft threshold={soft_threshold:.4f}"
        )

        filtered = [
            (chunk, score)
            for chunk, score in all_results
            if score >= soft_threshold
        ]

    # ─────────────────────────────────────────────────────────────
    # Sort descending
    # ─────────────────────────────────────────────────────────────
    filtered.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    # Delayed rounding (better ranking precision)
    final_results = [
        (chunk, round(score, 4))
        for chunk, score in filtered[:top_k]
    ]

    logger.info(
        f"[HybridRetriever] "
        f"query='{query[:80]}' "
        f"results={len(final_results)} "
        f"threshold={threshold:.4f} "
        f"semantic={semantic_available}"
    )

    return final_results


def build_context(
    ranked_chunks: List[Tuple[KnowledgeChunk, float]],
) -> str:
    """
    Build clean LLM context string.
    """

    if not ranked_chunks:
        return ""

    parts: List[str] = []

    for idx, (chunk, score) in enumerate(
        ranked_chunks,
        start=1,
    ):

        safe_category = getattr(
            chunk,
            "category",
            "unknown",
        )

        safe_text = getattr(
            chunk,
            "text",
            "",
        ).strip()

        if not safe_text:
            continue

        parts.append(
            
            
            
            f"{safe_text}"
        )

    return "\n\n".join(parts)