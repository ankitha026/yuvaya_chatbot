"""
greeting_handler.py – Lightweight conversational preprocessing.

Purpose:
- Detect greeting / thanks / goodbye / acknowledgement messages
- Strip conversational prefixes from mixed queries
- Avoid unnecessary retrieval + LLM calls
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from src.preprocessing.patterns import (
    ACKNOWLEDGEMENT_PATTERNS,
    ACKNOWLEDGEMENT_RESPONSES,
    GOODBYE_PATTERNS,
    GOODBYE_RESPONSES,
    GREETING_PATTERNS,
    GREETING_RESPONSES,
    THANKS_PATTERNS,
    THANKS_RESPONSES,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GreetingResult:
    is_greeting_only: bool
    cleaned_query: Optional[str]
    greeting_response: Optional[str]


class GreetingPreprocessor:
    """
    Handles:
    1. Greeting only
    2. Thanks only
    3. Goodbye only
    4. Acknowledgement only
    5. Greeting + query
    6. Normal query
    """

    def __init__(self) -> None:

        self.greeting_pattern = self._build_pattern(GREETING_PATTERNS)
        self.thanks_pattern = self._build_pattern(THANKS_PATTERNS)
        self.goodbye_pattern = self._build_pattern(GOODBYE_PATTERNS)
        self.ack_pattern = self._build_pattern(ACKNOWLEDGEMENT_PATTERNS)

    def _build_pattern(self, patterns: list[str]) -> re.Pattern:

        regex = "|".join(
            f"(?:{pattern})"
            for pattern in patterns
        )

        return re.compile(
            rf"""
            ^
            \s*
            ({regex})
            (?:
                [\s,.!?-]+
            )?
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def normalize(self, text: str) -> str:
        """
        Normalize repeated spaces.
        """

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text

    def _match_full(
        self,
        pattern: re.Pattern,
        text: str,
    ) -> bool:
        """
        Checks whether the ENTIRE message
        is only conversational text.
        """

        cleaned = re.sub(r"[.!?,]+$", "", text).strip()

        return bool(pattern.fullmatch(cleaned))

    def _strip_prefix(
        self,
        pattern: re.Pattern,
        text: str,
    ) -> str:

        match = pattern.match(text)

        if not match:
            return text

        return text[match.end():].strip()

    def process(self, user_input: str) -> GreetingResult:

        if not isinstance(user_input, str):
            return GreetingResult(
                is_greeting_only=False,
                cleaned_query="",
                greeting_response=None,
            )

        original = self.normalize(user_input)

        if not original:
            return GreetingResult(
                is_greeting_only=False,
                cleaned_query="",
                greeting_response=None,
            )

        lowered = original.lower()

        # ─────────────────────────────────────────────
        # Greeting-only
        # ─────────────────────────────────────────────
        if self._match_full(self.greeting_pattern, lowered):

            logger.info(
                "[GreetingPreprocessor] Greeting-only message detected"
            )

            return GreetingResult(
                is_greeting_only=True,
                cleaned_query=None,
                greeting_response=random.choice(
                    GREETING_RESPONSES
                ),
            )

        # ─────────────────────────────────────────────
        # Thanks-only
        # ─────────────────────────────────────────────
        if self._match_full(self.thanks_pattern, lowered):

            logger.info(
                "[GreetingPreprocessor] Thanks-only message detected"
            )

            return GreetingResult(
                is_greeting_only=True,
                cleaned_query=None,
                greeting_response=random.choice(
                    THANKS_RESPONSES
                ),
            )

        # ─────────────────────────────────────────────
        # Goodbye-only
        # ─────────────────────────────────────────────
        if self._match_full(self.goodbye_pattern, lowered):

            logger.info(
                "[GreetingPreprocessor] Goodbye-only message detected"
            )

            return GreetingResult(
                is_greeting_only=True,
                cleaned_query=None,
                greeting_response=random.choice(
                    GOODBYE_RESPONSES
                ),
            )

        # ─────────────────────────────────────────────
        # Acknowledgement-only
        # ─────────────────────────────────────────────
        if self._match_full(self.ack_pattern, lowered):

            logger.info(
                "[GreetingPreprocessor] Acknowledgement-only detected"
            )

            return GreetingResult(
                is_greeting_only=True,
                cleaned_query=None,
                greeting_response=random.choice(
                    ACKNOWLEDGEMENT_RESPONSES
                ),
            )

        # ─────────────────────────────────────────────
        # Greeting + query
        # Example:
        # "hello show collagen products"
        # ─────────────────────────────────────────────
        cleaned = self._strip_prefix(
            self.greeting_pattern,
            original,
        )

        if cleaned != original:

            logger.info(
                "[GreetingPreprocessor] Greeting stripped | original='%s' | cleaned='%s'",
                original,
                cleaned,
            )

            return GreetingResult(
                is_greeting_only=False,
                cleaned_query=cleaned,
                greeting_response=None,
            )

        # ─────────────────────────────────────────────
        # Normal query
        # ─────────────────────────────────────────────
        return GreetingResult(
            is_greeting_only=False,
            cleaned_query=original,
            greeting_response=None,
        )