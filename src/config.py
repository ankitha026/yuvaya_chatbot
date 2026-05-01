"""
config.py – Centralized application configuration.

Loads environment variables from .env and exposes validated runtime settings.

Responsibilities:
- Environment variable loading
- Runtime configuration validation
- Typed settings access
- Path management
- Secure defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"

# Load .env file
load_dotenv(ENV_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_env(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = os.getenv(key, default)

    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(
            f"Missing required environment variable: {key}"
        )

    return str(value) if value is not None else ""


def _get_int(
    key: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _get_env(key, str(default))

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid integer for {key}: {raw}"
        ) from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(f"{key} must be >= {minimum}")

    if maximum is not None and value > maximum:
        raise RuntimeError(f"{key} must be <= {maximum}")

    return value


def _get_float(
    key: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _get_env(key, str(default))

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid float for {key}: {raw}"
        ) from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(f"{key} must be >= {minimum}")

    if maximum is not None and value > maximum:
        raise RuntimeError(f"{key} must be <= {maximum}")

    return value


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

class Settings:
    """
    Typed runtime configuration container.
    """

    # ── App ──────────────────────────────────────────────────────────────────

    APP_NAME: str = "Collagreens Chatbot API"
    APP_VERSION: str = "1.0.0"

    ENVIRONMENT: str = _get_env("ENVIRONMENT", "development").lower()

    if ENVIRONMENT not in {"development", "production"}:
        raise RuntimeError("ENVIRONMENT must be 'development' or 'production'")

    DEBUG: bool = _get_env("DEBUG", "false").lower() == "true"

    # ── Redis ────────────────────────────────────────────────────────────────

    REDIS_HOST: str = _get_env("REDIS_HOST", "localhost")

    REDIS_PORT: int = _get_int(
        "REDIS_PORT",
        6379,
        minimum=1,
        maximum=65535,
    )

    SESSION_TTL: int = _get_int(
        "SESSION_TTL",
        1800,
        minimum=60,
        maximum=86400,
    )

    # ── LLM Provider ─────────────────────────────────────────────────────────

    LLM_PROVIDER: str = _get_env(
        "LLM_PROVIDER",
        "groq",
    ).lower()

    SUPPORTED_PROVIDERS = {
        "groq"
    }

    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
        )

    # ── API Keys ─────────────────────────────────────────────────────────────

    GROQ_API_KEY: str = _get_env("GROQ_API_KEY", "")

    _PROVIDER_KEY_MAP = {
        "groq": GROQ_API_KEY,
    }

    if ENVIRONMENT == "production" and not _PROVIDER_KEY_MAP.get(LLM_PROVIDER):
        raise RuntimeError(
            f"Missing API key for provider: {LLM_PROVIDER}"
        )

    # ── Models ───────────────────────────────────────────────────────────────

    GROQ_MODEL: str = _get_env(
        "GROQ_MODEL",
        "llama-3.1-8b-instant",
    )

    # ── Retrieval ────────────────────────────────────────────────────────────

    TOP_K_CHUNKS: int = _get_int(
        "TOP_K_CHUNKS",
        3,
        minimum=1,
        maximum=50,
    )

    SIMILARITY_THRESHOLD: float = _get_float(
        "SIMILARITY_THRESHOLD",
        0.30,
        minimum=0.0,
        maximum=1.0,
    )

    KEYWORD_WEIGHT: float = _get_float(
        "KEYWORD_WEIGHT",
        0.4,
        minimum=0.0,
        maximum=1.0,
    )

    SEMANTIC_WEIGHT: float = _get_float(
        "SEMANTIC_WEIGHT",
        0.6,
        minimum=0.0,
        maximum=1.0,
    )

    KEYWORD_FIELD_WEIGHT: float = _get_float(
        "KEYWORD_FIELD_WEIGHT",
        0.7,
        minimum=0.0,
        maximum=1.0,
    )

    TEXT_FIELD_WEIGHT: float = _get_float(
        "TEXT_FIELD_WEIGHT",
        0.3,
        minimum=0.0,
        maximum=1.0,
    )

    KEYWORD_EXACT_MATCH_BOOST: float = _get_float(
        "KEYWORD_EXACT_MATCH_BOOST",
        0.2,
        minimum=0.0,
        maximum=1.0,
    )

    # Hybrid Retrieval

    SOFT_THRESHOLD_FACTOR: float = _get_float(
        "SOFT_THRESHOLD_FACTOR",
        0.7,
        minimum=0.0,
        maximum=1.0,
    )

    RETRIEVAL_CANDIDATE_MULTIPLIER: int = _get_int(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        3,
        minimum=1,
        maximum=10,
    )

    MAX_RETRIEVAL_LIMIT: int = _get_int(
        "MAX_RETRIEVAL_LIMIT",
        50,
        minimum=1,
        maximum=200,
    )

    # ── Embeddings ───────────────────────────────────────────────────────────

    EMBEDDING_MODEL_NAME: str = _get_env(
        "EMBEDDING_MODEL_NAME",
        "all-MiniLM-L6-v2",
    )

    EMBEDDING_BATCH_SIZE: int = _get_int(
        "EMBEDDING_BATCH_SIZE",
        32,
        minimum=1,
        maximum=256,
    )

    EMBEDDING_CACHE_ENABLED: bool = (
        _get_env("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )

    BASE_DIR: Path = BASE_DIR

    # Validate retrieval weights safely
    if abs((KEYWORD_WEIGHT + SEMANTIC_WEIGHT) - 1.0) > 1e-6:
        raise RuntimeError(
            "KEYWORD_WEIGHT + SEMANTIC_WEIGHT must equal 1.0"
        )

    # ── Memory ───────────────────────────────────────────────────────────────

    MEMORY_WINDOW: int = _get_int(
        "MEMORY_WINDOW",
        3,
        minimum=1,
        maximum=50,
    )

    # ── Server ───────────────────────────────────────────────────────────────

    HOST: str = _get_env("HOST", "0.0.0.0")

    PORT: int = _get_int(
        "PORT",
        8000,
        minimum=1,
        maximum=65535,
    )

    LOG_LEVEL: str = _get_env(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    # ── Logging / Security ───────────────────────────────────────────────────

    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ENABLE_LOG_MASKING: bool = (
        _get_env("ENABLE_LOG_MASKING", "true").lower() == "true"
    )

    # ── Data ─────────────────────────────────────────────────────────────────

    DATA_DIR: Path = BASE_DIR / "src" / "data"

    # ── Rate Limiting ────────────────────────────────────────────────────────

    RATE_LIMIT_PER_MINUTE: int = _get_int(
        "RATE_LIMIT_PER_MINUTE",
        30,
        minimum=1,
        maximum=1000,
    )

    # ── Timeouts ─────────────────────────────────────────────────────────────

    LLM_TIMEOUT_SECONDS: int = _get_int(
        "LLM_TIMEOUT_SECONDS",
        30,
        minimum=5,
        maximum=300,
    )


# Singleton settings instance
settings = Settings()