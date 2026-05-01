"""
query_rewriter.py – Lightweight query normalization and sanitization.

Purpose:
- Clean user queries before retrieval
- Improve retrieval consistency
- Prevent noisy or malformed input from affecting search quality

Design goals:
- Fast (rule-based only)
- Safe
- Deterministic
- Low latency
"""
from __future__ import annotations

import logging
import re

from langdetect import detect

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en"}


def _is_supported_language(text: str) -> bool:
    """
    Detect whether the query language is supported.
    Defaults to True if detection fails.
    """
    try:
        lang = detect(text)
        return lang in SUPPORTED_LANGUAGES
    except Exception:
        logger.warning("[QueryRewriter] Language detection failed.")
        return True

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 1000

# ── Common contractions and phrase rewrites ──────────────────────────────────
_REPLACEMENTS = {
    r"\bwhat's\b": "what is",
    r"\bwhat're\b": "what are",
    r"\bhow's\b": "how is",
    r"\bit's\b": "it is",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'd\b": "i would",
    r"\bwhat about\b": "tell me about",
    r"\btell me\b": "what is",
}

# ── Noise / filler patterns ──────────────────────────────────────────────────
_FILLER_PATTERNS = [
    r"\bplease\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bi want to know\b",
    r"\bi would like to know\b",
    r"\bkindly\b",
]

# ── Unsafe / unwanted characters ─────────────────────────────────────────────
_UNSAFE_PATTERN = re.compile(r"[^\w\s?.!,%-]")

# ── Multi-space cleanup ──────────────────────────────────────────────────────
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def rewrite_query(query: str) -> str:
    """
    Normalize and lightly rewrite a user query for better retrieval.

    Steps:
    1. Validate input
    2. Trim and lowercase
    3. Remove unsafe characters
    4. Expand contractions
    5. Remove filler phrases
    6. Normalize whitespace
    7. Remove trailing punctuation

    Returns:
        Cleaned query string
    """

    # ── Input validation ────────────────────────────────────────────────────
    if not isinstance(query, str):
        logger.warning(
            "[QueryRewriter] Non-string query received. type=%s",
            type(query).__name__,
        )
        return ""

    # ── Trim oversized input ───────────────────────────────────────────────
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(
            "[QueryRewriter] Query exceeded max length (%s). Truncating.",
            MAX_QUERY_LENGTH,
        )
        query = query[:MAX_QUERY_LENGTH]

    # ── Normalize case and whitespace ──────────────────────────────────────
    q = query.strip().lower()

    if not q:
        return ""
    
    # ── Language detection ─────────────────────────────────────────────
    if not _is_supported_language(q):
        logger.info("[QueryRewriter] Unsupported language detected.")
        return (
            "Please ask your question in English so I can assist you accurately."
        )

    # ── Remove potentially unsafe characters ───────────────────────────────
    q = _UNSAFE_PATTERN.sub(" ", q)

    # ── Expand contractions / rewrites ─────────────────────────────────────
    for pattern, replacement in _REPLACEMENTS.items():
        q = re.sub(pattern, replacement, q)

    # ── Remove conversational filler ───────────────────────────────────────
    for pattern in _FILLER_PATTERNS:
        q = re.sub(pattern, " ", q)

    # ── Normalize repeated whitespace ──────────────────────────────────────
    q = _MULTI_SPACE_PATTERN.sub(" ", q).strip()

    # ── Remove trailing punctuation ────────────────────────────────────────
    q = re.sub(r"[?.!]+$", "", q).strip()

    logger.debug("[QueryRewriter] Original='%s' | Rewritten='%s'", query, q)

    return q