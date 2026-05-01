"""
logger.py – Structured logging for the Collagreens chatbot (production-ready).

Features:
  - Console + rotating file logging
  - Structured JSON logging for chat events
  - Sensitive data masking (emails, phone numbers, API keys)
  - Config-driven masking toggle (enable/disable)
  - Safe logging (never crashes the app)
"""

from __future__ import annotations
import json
import logging
import logging.handlers
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.config import settings
from src.data.knowledge_base import KnowledgeChunk


# ─────────────────────────────────────────────────────────────────────────────
# Masking Configuration
# ─────────────────────────────────────────────────────────────────────────────

MASKING_ENABLED = settings.ENABLE_LOG_MASKING  # You can later move this to settings if needed


def mask_sensitive_data(text: str) -> str:
    """
    Masks sensitive user data in logs.
    Covers:
      - Emails
      - Phone numbers (basic 10-digit)
      - API keys / tokens (basic patterns)
    """
    if not text or not MASKING_ENABLED:
        return text

    try:
        # Mask emails
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL_MASKED]', text)

        # Mask phone numbers (10-digit)
        text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)

        # Mask API keys / tokens
        text = re.sub(r'(sk-|api_key=|token=)[\w-]+', '[KEY_MASKED]', text, flags=re.IGNORECASE)

    except Exception:
        # Never break logging due to masking failure
        return text

    return text

class MaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Mask main message
            if isinstance(record.msg, str):
                record.msg = mask_sensitive_data(record.msg)

            # 🔥 IMPORTANT: Mask arguments also
            if record.args:
                safe_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        safe_args.append(mask_sensitive_data(arg))
                    else:
                        safe_args.append(arg)
                record.args = tuple(safe_args)

        except Exception:
            pass

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Logger Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger with console + rotating file handler."""

    log_dir: Path = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chatbot.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()

    # Prevent duplicate handlers (important in reload/dev environments)
    if root.handlers:
        root.handlers.clear()

    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    #Create filter
    
    mask_filter = MaskingFilter()

    # Console handler (less verbose)
    #ch = logging.StreamHandler()
    #ch.setLevel(logging.WARNING)
    #ch.setFormatter(fmt)
    #ch.addFilter(mask_filter)
    #root.addHandler(ch)

    # Rotating file handler (persistent logs)
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(mask_filter)
    root.addHandler(fh)


# Dedicated logger for chatbot events
_chat_logger = logging.getLogger("chatbot.chat")

# Reduce noisy third-party logs and prevent formatting conflicts
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Chat Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_query_event(
    session_id: str,
    query: str,
    enriched_query: str,
    ranked_chunks: List[Tuple[KnowledgeChunk, float]],
    llm_response: str,
    is_fallback: bool,
) -> None:
    """
    Log a single chat interaction as structured JSON.
    Safe, masked, and production-friendly.
    """
    try:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "query": mask_sensitive_data(query),
            "enriched_query": mask_sensitive_data(enriched_query),
            "is_fallback": is_fallback,
            "retrieved_chunks": [
                {
                    "id": c.id,
                    "category": c.category,
                    "score": score
                }
                for c, score in ranked_chunks
            ],
            "llm_response_preview": mask_sensitive_data(llm_response[:200]),
        }

        _chat_logger.info(json.dumps(event, ensure_ascii=False))

    except Exception as exc:
        # Logging should NEVER crash the application
        logging.getLogger(__name__).error(
            f"[Logger] Failed to log query event: {exc}",
            exc_info=True
        )