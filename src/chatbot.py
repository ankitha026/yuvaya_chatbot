"""
chatbot.py – Core chatbot orchestration.

Pipeline:
  1. Enrich query with conversation history (for follow-ups)
  2. Rewrite / normalise query
  3. Hybrid retrieval (keyword + semantic)
  4. Confidence check → fallback if weak context
  5. Build context string
  6. Call LLM with full message history + context
  7. Log interaction
  8. Store response in memory
  9. Return reply
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from src.config import settings
from src.retrieval.hybrid_retriever import hybrid_retrieve, build_context
from src.llm.llm_interface import generate_response, FALLBACK_RESPONSE
from src.memory.memory import memory
from src.utils.logger import log_query_event
from src.utils.query_rewriter import rewrite_query
from src.preprocessing.conversation_handler import GreetingPreprocessor


logger = logging.getLogger(__name__)

CHATBOT_COMPONENT = "chatbot_engine"


class ChatbotEngine:
    """
    Stateless orchestration class.

    Session state is maintained separately in the global
    ConversationMemory store.
    """

    def __init__(
        self,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ):
        self.top_k = top_k or settings.TOP_K_CHUNKS

        self.greeting_preprocessor = GreetingPreprocessor()

        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not (0 <= self.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1")

    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> str:

        start_time = time.perf_counter()

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        preprocessed = self.greeting_preprocessor.process(query)

        #Conversational shortcut

        if preprocessed.is_greeting_only:
            logger.info(f"[{CHATBOT_COMPONENT}] Conversational shortcut response returned")
            return preprocessed.greeting_response
        
        query = preprocessed.cleaned_query

        if not query:
            raise ValueError("query cannot be empty")

        if len(query) > 2000:
            raise ValueError("query exceeds maximum allowed length")

        session_id = session_id or str(uuid.uuid4())

        logger.info(
            f"[{CHATBOT_COMPONENT}] Processing chat request | session_id={session_id}"
        )

        ranked_chunks = []
        is_fallback = False

        try:
            # 1. Enrich query
            enriched_query = memory.enrich_query_with_context(
                session_id=session_id,
                query=query,
            )

            # 2. Rewrite query
            clean_query = rewrite_query(enriched_query)

            # 3. Retrieval
            ranked_chunks = hybrid_retrieve(
                query=clean_query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
            )

            # 4. Fallback check
            if not ranked_chunks:
                is_fallback = True
                reply = FALLBACK_RESPONSE

            else:
                top_score = ranked_chunks[0][1]
                avg_score = sum(score for _, score in ranked_chunks) / len(ranked_chunks) if ranked_chunks else 0

                context = build_context(ranked_chunks)

                relevant = any(
                    c.category in ["support","returns_policy","shipping_policy","sales_purchase","quality_trust","ingredients","benefits","usage","product_overview"]
                    for c, _ in ranked_chunks
                )
    
                if (
                    top_score < 0.55
                    or avg_score < 0.45
                    or not relevant
                    or "collagreens" not in context.lower() 
                  ):
                    is_fallback = True
                    reply = FALLBACK_RESPONSE

                else:
                    # 5. Build context
                    context = build_context(ranked_chunks)

                    if not context.strip():
                        is_fallback = True
                        reply = FALLBACK_RESPONSE

                    else:
                        # 6. Memory (FIXED SAFE BLOCK)
                        try:
                            messages = memory.build_messages_for_llm(
                                session_id=session_id,
                                current_query=query,
                            )

                            if not messages:
                                raise ValueError("Empty memory")

                        except Exception:
                            logger.warning(
                                f"[{CHATBOT_COMPONENT}] Memory unavailable, using stateless mode"
                            )

                            messages = [
                                {"role": "user", "content": query}
                            ]

                        # 7. LLM call
                        reply = generate_response(
                            messages=messages,
                            context=context,
                        )

                        if not reply or not reply.strip():
                            is_fallback = True
                            reply = FALLBACK_RESPONSE

            # 8. Logging
            log_query_event(
                session_id=session_id,
                query=query,
                enriched_query=enriched_query,
                ranked_chunks=ranked_chunks,
                llm_response=reply,
                is_fallback=is_fallback,
            )

            # 9. Memory storage
            memory.add_user(session_id, query)
            memory.add_assistant(session_id, reply)

            elapsed = round(time.perf_counter() - start_time, 4)

            logger.info(
                f"[{CHATBOT_COMPONENT}] Request completed in {elapsed}s | session_id={session_id}"
            )

            return reply

        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 4)

            logger.exception(
                f"[{CHATBOT_COMPONENT}] Chat failed after {elapsed}s | session_id={session_id} | error={exc}"
            )

            raise


engine = ChatbotEngine()

__all__ = ["ChatbotEngine", "engine"]