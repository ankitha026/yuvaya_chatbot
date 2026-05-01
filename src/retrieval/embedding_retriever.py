"""
embedding_retriever.py - Production-grade semantic retrieval using
sentence-transformers + FAISS.

Features:
- FAISS vector index
- Thread-safe singleton initialization
- Persistent index caching
- Automatic cache invalidation
- Embedding normalization
- Batch encoding
- Startup warmup support
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.data.knowledge_base import KnowledgeChunk, get_all_chunks

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_indexed_chunks: List[KnowledgeChunk] = []

_lock = threading.Lock()

# ── Cache paths ───────────────────────────────────────────────────────────────
EMBEDDING_CACHE_DIR = settings.BASE_DIR / "cache"
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = EMBEDDING_CACHE_DIR / "semantic.index"
CHUNK_CACHE_PATH = EMBEDDING_CACHE_DIR / "indexed_chunks.pkl"
VERSION_FILE_PATH = EMBEDDING_CACHE_DIR / "index_version.txt"

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = settings.EMBEDDING_MODEL_NAME


# ── Model loader ──────────────────────────────────────────────────────────────
def _get_model() -> SentenceTransformer:
    """
    Lazy-load embedding model once.
    """
    global _model

    if _model is None:
        try:
            logger.info(
                "[EmbeddingRetriever] Loading embedding model: %s",
                MODEL_NAME,
            )

            _model = SentenceTransformer(MODEL_NAME)

            logger.info(
                "[EmbeddingRetriever] Embedding model loaded successfully."
            )

        except Exception as exc:
            logger.exception(
                "[EmbeddingRetriever] Failed to load embedding model."
            )
            raise RuntimeError(
                "Failed to initialize embedding model."
            ) from exc

    return _model


# ── Warmup ────────────────────────────────────────────────────────────────────
def warmup_model() -> None:
    """
    Warm up embedding model during startup to avoid first-request latency.
    """
    try:
        model = _get_model()

        model.encode(
            ["warmup query"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        logger.info("[EmbeddingRetriever] Model warmup completed.")

    except Exception:
        logger.exception("[EmbeddingRetriever] Model warmup failed.")


# ── Embedding helpers ─────────────────────────────────────────────────────────
def _encode_texts(texts: List[str]) -> np.ndarray:
    """
    Generate normalized embeddings.
    """
    model = _get_model()

    embeddings = model.encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def _compute_chunks_version(chunks: List[KnowledgeChunk]) -> str:
    """
    Compute stable content hash for cache invalidation.
    """
    payload = [
        {
            "id": c.id,
            "text": c.text,
            "category": c.category,
        }
        for c in chunks
    ]

    raw = json.dumps(payload, sort_keys=True).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ── Persistence ───────────────────────────────────────────────────────────────
def _save_index(index: faiss.Index, chunks: List[KnowledgeChunk]) -> None:
    """
    Persist FAISS index and metadata.
    """
    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        with open(CHUNK_CACHE_PATH, "wb") as f:
            pickle.dump(chunks, f)

        version = _compute_chunks_version(chunks)
        VERSION_FILE_PATH.write_text(version)

        logger.info("[EmbeddingRetriever] Saved FAISS index to disk.")

    except Exception:
        logger.exception(
            "[EmbeddingRetriever] Failed to save FAISS index."
        )


def _load_index(chunks: List[KnowledgeChunk]) -> bool:
    """
    Load cached FAISS index if valid.
    """
    global _index, _indexed_chunks

    try:
        if not FAISS_INDEX_PATH.exists():
            return False

        if not CHUNK_CACHE_PATH.exists():
            return False

        if not VERSION_FILE_PATH.exists():
            return False

        current_version = _compute_chunks_version(chunks)
        cached_version = VERSION_FILE_PATH.read_text().strip()

        if current_version != cached_version:
            logger.info(
                "[EmbeddingRetriever] Knowledge base changed. Rebuilding index."
            )
            return False

        _index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNK_CACHE_PATH, "rb") as f:
            _indexed_chunks = pickle.load(f)

        logger.info(
            "[EmbeddingRetriever] Loaded cached semantic index (%s chunks).",
            len(_indexed_chunks),
        )

        return True

    except Exception:
        logger.exception(
            "[EmbeddingRetriever] Failed to load cached index."
        )
        return False


# ── Index builder ─────────────────────────────────────────────────────────────
def _build_index(chunks: List[KnowledgeChunk]) -> None:
    """
    Build FAISS semantic index.
    """
    global _index, _indexed_chunks

    logger.info(
        "[EmbeddingRetriever] Building semantic index for %s chunks...",
        len(chunks),
    )

    texts = [chunk.text for chunk in chunks]

    embeddings = _encode_texts(texts)

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    _index = index
    _indexed_chunks = chunks

    logger.info(
        "[EmbeddingRetriever] Semantic FAISS index ready."
    )

    _save_index(index, chunks)


# ── Public initializer ────────────────────────────────────────────────────────
def ensure_index(chunks: List[KnowledgeChunk] | None = None) -> None:
    """
    Ensure semantic index exists.
    Thread-safe singleton initialization.
    """
    global _index

    # Fast path
    if _index is not None:
        return

    with _lock:

        # Double-check after acquiring lock
        if _index is not None:
            return

        if chunks is None:
            chunks = get_all_chunks()

        if _load_index(chunks):
            return

        _build_index(chunks)


# ── Semantic retrieval ────────────────────────────────────────────────────────
def semantic_retrieve(
    query: str,
    chunks: List[KnowledgeChunk] | None = None,
    top_k: int = 10,
) -> List[Tuple[KnowledgeChunk, float]]:
    """
    Fast semantic retrieval using FAISS.
    """

    if not isinstance(query, str):
        logger.warning(
            "[EmbeddingRetriever] Non-string query received."
        )
        return []

    query = query.strip()

    if not query:
        return []

    try:
        ensure_index(chunks)

        if _index is None:
            logger.error(
                "[EmbeddingRetriever] Semantic index unavailable."
            )
            return []

        query_embedding = _encode_texts([query])

        scores, indices = _index.search(query_embedding, top_k)

        results: List[Tuple[KnowledgeChunk, float]] = []

        for idx, score in zip(indices[0], scores[0]):

            if idx < 0:
                continue

            if idx >= len(_indexed_chunks):
                continue

            chunk = _indexed_chunks[idx]

            results.append(
                (chunk, round(float(score), 4))
            )

        logger.debug(
            "[EmbeddingRetriever] Retrieved %s semantic chunks.",
            len(results),
        )

        return results

    except Exception:
        logger.exception(
            "[EmbeddingRetriever] Semantic retrieval failed."
        )
        return []