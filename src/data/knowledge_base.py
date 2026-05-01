"""
knowledge_base.py – Real product knowledge base for Collagreens by Yuvaya.
Data source: Official Chatbot_questions.docx (processed and enriched).
Sections covered:
  1. Product Basics        5. Sales & Purchase
  2. Usage & Results       6. Shipping & Delivery
  3. Ingredient Trust      7. Returns, Refunds & Cancellations
  4. Objections & Trust    8. Product-Related Queries / Support
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List, Optional

logger = logging.getLogger(__name__)

# Configurable data file path (env override supported)
DATA_FILE = Path(os.getenv("KB_DATA_FILE", Path(__file__).parent / "collagreens_data.json"))

# Thread-safe cache
_cache: Optional[List["KnowledgeChunk"]] = None
_cache_lock = Lock()


@dataclass
class KnowledgeChunk:
    id: str
    category: str
    text: str
    keywords: List[str] = field(default_factory=list)


def _validate_chunk(raw: dict, index: int) -> Optional[KnowledgeChunk]:
    """Validate and clean a single chunk."""
    required = ("id", "category", "text", "keywords")

    for f in required:
        if not raw.get(f):
            logger.warning(f"[KB] Skipping chunk at index {index} — missing '{f}'")
            return None

    if not isinstance(raw["keywords"], list):
        logger.warning(f"[KB] Skipping '{raw.get('id', '?')}' — keywords not a list")
        return None

    if len(raw["text"].strip()) < 20:
        logger.warning(f"[KB] Skipping '{raw['id']}' — text too short")
        return None

    cleaned_keywords = [kw.strip().lower() for kw in raw["keywords"] if isinstance(kw, str) and kw.strip()]

    if len(cleaned_keywords) < 2:
        logger.warning(f"[KB] Skipping '{raw['id']}' — fewer than 2 valid keywords")
        return None

    return KnowledgeChunk(
        id=raw["id"],
        category=raw["category"],
        text=raw["text"].strip(),
        keywords=cleaned_keywords,
    )


def _load_from_json(path: Path) -> List[KnowledgeChunk]:
    """Load and validate all chunks from JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run scripts/preprocess_data.py to regenerate it."
        )

    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw_list = json.load(fh)
        except json.JSONDecodeError as e:
            logger.error(f"[KB] Failed to parse JSON: {e}")
            raise

    if not isinstance(raw_list, list):
        raise ValueError("[KB] JSON root must be a list of chunks")

    chunks: List[KnowledgeChunk] = []
    seen_ids = set()

    invalid_count = 0
    duplicate_count = 0

    for idx, raw in enumerate(raw_list):
        chunk = _validate_chunk(raw, idx)
        if chunk is None:
            invalid_count += 1
            continue

        if chunk.id in seen_ids:
            logger.warning(f"[KB] Duplicate ID '{chunk.id}' — skipping")
            duplicate_count += 1
            continue

        seen_ids.add(chunk.id)
        chunks.append(chunk)

    logger.info(
        f"[KB] Loaded {len(chunks)} valid chunks | "
        f"Dropped {invalid_count} invalid | "
        f"Skipped {duplicate_count} duplicates"
    )

    if not chunks:
        logger.error("[KB] No valid knowledge chunks loaded — system may fail")

    return chunks


def get_all_chunks() -> List[KnowledgeChunk]:
    """Thread-safe access to cached knowledge chunks."""
    global _cache

    if _cache is None:
        with _cache_lock:
            if _cache is None:  # Double-checked locking
                logger.info("[KB] Cache miss — loading data from disk")
                _cache = _load_from_json(DATA_FILE)

    return _cache


def reload_chunks() -> List[KnowledgeChunk]:
    """Force reload of knowledge base (useful for admin/debug)."""
    global _cache

    with _cache_lock:
        logger.info("[KB] Reloading knowledge base...")
        _cache = _load_from_json(DATA_FILE)

    return _cache


# Lazy load instead of eager load (safer for large systems)
def init_knowledge_base():
    """Optional explicit initializer (recommended for app startup)."""
    get_all_chunks()

