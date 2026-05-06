"""
memory.py – Redis-backed short-term conversation memory manager.

Maintains a rolling window of the last N user/assistant turns per session.
Used to provide context for follow-up questions.

Storage:
- Redis
- JSON serialized message lists
- TTL-based automatic cleanup
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import redis
import time

from src.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Redis-backed conversation memory store.

    Each session_id stores:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    def __init__(self, window: int | None = None):
        self.window = window or settings.MEMORY_WINDOW

        # Max stored messages
        # window=3 => max 6 messages
        self.max_messages = self.window * 2

        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.session_ttl = int(os.getenv("SESSION_TTL", "1800"))

        self.redis_client: Optional[redis.Redis] = None

        self._initialize_redis()

    # ─────────────────────────────────────────────────────────────
    # Redis Setup
    # ─────────────────────────────────────────────────────────────

    def _initialize_redis(self) -> None:
        """
        Initialize Redis connection with health check.
        """
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )

            for attempt in range(5):
                try:
                    self.redis_client.ping()
                    logger.info(
                        "[Memory] Redis connected successfully "
                        f"({self.redis_host}:{self.redis_port})"
                    )
                    return
                except Exception:
                    logger.warning(f"[Memory] Redis not ready, retrying...({attempt+1}/5)")
                    time.sleep(2)
            logger.warning("[Memory] Redis connection failed after multiple attempts.")
            
            self.redis_client = None
                    

        except Exception as exc:
            logger.exception(
                f"[Memory] Failed to initialize Redis connection: {exc}"
            )
            self.redis_client = None

    # ─────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────

    def _session_key(self, session_id: str) -> str:
        """
        Generate Redis key for session.
        """
        return f"chat_memory:{session_id}"

    def _load_messages(self, session_id: str) -> List[Dict]:
        """
        Load message history from Redis.
        """
        if not self.redis_client:
            logger.warning("[Memory] Redis unavailable during read.")
            return []

        try:
            raw = self.redis_client.get(self._session_key(session_id))

            if not raw:
                return []

            messages = json.loads(raw)

            if not isinstance(messages, list):
                logger.warning(
                    f"[Memory] Invalid message format for session={session_id}"
                )
                return []

            return messages

        except Exception as exc:
            logger.exception(
                f"[Memory] Failed loading session={session_id}: {exc}"
            )
            return []

    def _save_messages(self, session_id: str, messages: List[Dict]) -> None:
        """
        Save messages to Redis with TTL.
        """
        if not self.redis_client:
            logger.warning("[Memory] Redis unavailable during write.")
            return

        try:
            # Keep only latest N messages
            trimmed = messages[-self.max_messages:]

            self.redis_client.setex(
                name=self._session_key(session_id),
                time=self.session_ttl,
                value=json.dumps(trimmed),
            )

        except Exception as exc:
            logger.exception(
                f"[Memory] Failed saving session={session_id}: {exc}"
            )

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def add_user(self, session_id: str, message: str) -> None:
        """
        Add user message.
        """
        self.add_message(
            session_id,
            {"role": "user", "content": message},
        )

    def add_assistant(self, session_id: str, message: str) -> None:
        """
        Add assistant message.
        """
        self.add_message(
            session_id,
            {"role": "assistant", "content": message},
        )

    def add_message(self, session_id: str, message: Dict) -> None:
        """
        Add a message to conversation history.
        """
        messages = self._load_messages(session_id)
        messages.append(message)
        self._save_messages(session_id, messages)

    def get_history(self, session_id: str) -> List[Dict]:
        """
        Return conversation history for session.
        """
        return self._load_messages(session_id)

    def get_messages(self, session_id: str) -> List[Dict]:
        """
        Alias for compatibility.
        """
        return self.get_history(session_id)

    def build_messages_for_llm(
        self,
        session_id: str,
        current_query: str,
    ) -> List[Dict]:
        """
        Return:
        [history..., current_user_message]
        """
        history = self.get_history(session_id)

        return history + [
            {
                "role": "user",
                "content": current_query,
            }
        ]

    def clear(self, session_id: str) -> None:
        """
        Clear conversation history.
        """
        if not self.redis_client:
            return

        try:
            self.redis_client.delete(self._session_key(session_id))

        except Exception as exc:
            logger.exception(
                f"[Memory] Failed clearing session={session_id}: {exc}"
            )

    def enrich_query_with_context(
        self,
        session_id: str,
        query: str,
    ) -> str:
        """
        If query is very short, prepend previous user context.
        """
        SHORT_QUERY_THRESHOLD = 20

        if len(query.strip()) > SHORT_QUERY_THRESHOLD:
            return query

        history = self.get_history(session_id)

        recent_user_msgs = [
            m["content"]
            for m in history
            if m.get("role") == "user"
        ]

        if recent_user_msgs:
            return f"{recent_user_msgs[-1]} {query}"

        return query


# Global singleton
memory = ConversationMemory()