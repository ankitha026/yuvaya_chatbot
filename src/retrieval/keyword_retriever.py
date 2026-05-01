"""
keyword_retriever.py 

Improvements:
- Stopword removal
- Stemming normalization
- Exact phrase boost
- Stronger keyword weighting
- Safer scoring
- Better token normalization
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from nltk.stem import PorterStemmer

from src.config import settings
from src.data.knowledge_base import KnowledgeChunk, get_all_chunks

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# NLP Helpers
# ─────────────────────────────────────────────────────────────

_STEMMER = PorterStemmer()

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "of", "to", "and",
    "or", "for", "on", "at", "be", "do", "we", "you",
    "i", "my", "your", "our", "this", "that", "are",
    "has", "have", "with", "from", "by", "not", "no",
    "can", "will", "if", "as", "but", "so", "its",
    "was", "were", "they", "their", "them", "all",
    "any", "may", "should", "would", "could", "does",
    "did", "just", "also", "than", "then", "when",
    "what", "how", "why", "who", "which", "where",
}


def _normalize_token(token: str) -> str:
    """
    Lowercase + stemming normalization.
    """

    token = token.lower().strip()

    if not token:
        return ""

    return _STEMMER.stem(token)


def _tokenise(text: str) -> set[str]:
    """
    Production-grade tokenization.

    Features:
    - lowercase normalization
    - preserves useful alphanumeric tokens
    - removes punctuation noise
    - stopword filtering
    - stemming
    - avoids broken short tokens
    """

    if not text:
        return set()

    # Better token extraction
    raw_tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)?",
        text.lower(),
    )

    cleaned = set()

    for token in raw_tokens:

        token = token.strip("-_ ")

        if not token:
            continue

        # Remove pure numeric noise
        if token.isdigit():
            continue

        # Remove stopwords
        if token in _STOPWORDS:
            continue

        normalized = _normalize_token(token)

        # Ignore tiny/broken tokens
        if len(normalized) < 2:
            continue

        cleaned.add(normalized)

    return cleaned


def _exact_match_boost(
    query: str,
    chunk: KnowledgeChunk,
) -> float:
    """
    Extra score boost for exact phrase matches.
    """

    query_clean = query.lower().strip()

    if not query_clean:
        return 0.0

    keyword_text = " ".join(chunk.keywords).lower()
    chunk_text = chunk.text.lower()

    boost = 0.0

    if query_clean in keyword_text:
        boost += settings.KEYWORD_EXACT_MATCH_BOOST

    elif query_clean in chunk_text:
        boost += (
            settings.KEYWORD_EXACT_MATCH_BOOST * 0.5
        )

    return boost


def keyword_retrieve(
    query: str,
    chunks: List[KnowledgeChunk] | None = None,
    top_k: int = 10,
) -> List[Tuple[KnowledgeChunk, float]]:
    """
    Production keyword retrieval.

    Improvements:
    - stronger keyword weighting
    - stemming
    - stopword removal
    - exact match boosting
    """

    if chunks is None:
        chunks = get_all_chunks()

    if not query or not query.strip():
        logger.warning(
            "[KeywordRetriever] Empty query received."
        )
        return []

    # Safe top_k
    top_k = max(1, min(top_k, settings.MAX_RETRIEVAL_LIMIT))

    query_tokens = _tokenise(query)

    if not query_tokens:
        logger.warning(
            "[KeywordRetriever] Query became empty after normalization."
        )
        return []

    results: List[Tuple[KnowledgeChunk, float]] = []

    for chunk in chunks:

        # Safe handling
        if not getattr(chunk, "id", None):
            continue

        keyword_tokens = {
            _normalize_token(kw)
            for kw in chunk.keywords
            if kw
        }

        text_tokens = _tokenise(chunk.text)

        if not keyword_tokens and not text_tokens:
            continue

        # ─────────────────────────────────────────────────────
        # Matching
        # ─────────────────────────────────────────────────────
        keyword_matches = (
            query_tokens & keyword_tokens
        )

        text_matches = (
            query_tokens & text_tokens
        )

        # ─────────────────────────────────────────────────────
        # Scoring
        # ─────────────────────────────────────────────────────
        keyword_score = (
            len(keyword_matches)
            / max(len(keyword_tokens), 1)
        )

        text_score = (
            len(text_matches)
            / max(len(query_tokens), 1)
        )

        # Stronger keyword importance
        score = (
            settings.KEYWORD_FIELD_WEIGHT * keyword_score
            + settings.TEXT_FIELD_WEIGHT * text_score
        )

        # Exact match boost
        score += _exact_match_boost(
            query=query,
            chunk=chunk,
        )

        # Prevent overflow
        score = min(score, 1.0)

        if score > 0:
            results.append(
                (
                    chunk,
                    score,
                )
            )

    # Sort descending
    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    final_results = [
        (chunk, round(score, 4))
        for chunk, score in results[:top_k]
    ]

    logger.info(
        f"[KeywordRetriever] "
        f"query='{query[:80]}' "
        f"results={len(final_results)}"
    )

    return final_results