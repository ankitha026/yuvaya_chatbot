from __future__ import annotations
import logging
from typing import List, Dict
 
from groq import Groq, RateLimitError
from src.config import settings
 
logger = logging.getLogger(__name__)
 
# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly product assistant for Yuvaya's Collagreens.premium wellness supplement. Your job is to help customers understand the product, its benefits, ingredients, and science.
 
STRICT RULES:
1. Answer ONLY using provided context. Do NOT use outside knowledge or make up information.
2. If the context does not contain enough information to answer the question, say: "I don't have that information right now. For specific queries, please reach out to our support team at hello@yuvaya.in"
3.  Keep responses clear, concise, and natural — 2–4 sentences unless a detailed explanation is needed.
4. Be warm and slightly persuasive — you represent the brand.
5. Never guess, hallucinate, or invent facts about the product.
6. If the user asks something completely unrelated to Collagreens or wellness, politely redirect them.
"""
 
FALLBACK_RESPONSE = (
    "I'm sorry, I don't have enough information to answer that accurately.\n"
    "For the most up-to-date details, please reach out to our team:\n"
    "🌐hello@yuvaya.in\n"
    "We're happy to help!"
)
 
# ── CHANGE 1: Added rate limit response ───────────────────────
RATE_LIMIT_RESPONSE = (
    "Our assistant is experiencing very high demand right now. ⏳\n"
    "Please try again in a moment, or reach out to us directly:\n"
    "🌐hello@yuvaya.in\n"
    "We're happy to help!"
)
 
# ── Singleton Groq client ─────────────────────────────────────
_groq_client = Groq(api_key=settings.GROQ_API_KEY)
 
 
# ── Helpers ───────────────────────────────────────────────────
def _augment_messages_with_context(messages: List[Dict], context: str) -> List[Dict]:
    if not messages:
        return messages
 
    augmented = list(messages)
 
    # Inject context into last user message
    last = augmented[-1]
    if last["role"] == "user":
        augmented[-1] = {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{last['content']}"
            ),
        }
 
    return augmented
 
 
# ── Groq Call ─────────────────────────────────────────────────
def _call_groq(messages: List[Dict], context: str) -> str:
    augmented = _augment_messages_with_context(messages, context)
 
    # Add system prompt properly
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + augmented
 
    response = _groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=final_messages,
        temperature=0.3,
    )
 
    return response.choices[0].message.content.strip()
 
 
# ── Public API ────────────────────────────────────────────────
def generate_response(messages: List[Dict], context: str) -> str:
 
    if not context.strip():
        logger.info("[LLM] Empty context → fallback")
        return FALLBACK_RESPONSE
 
    try:
        return _call_groq(messages, context)
 
    # ── CHANGE 2: Catch rate limit before the generic handler ─
    except RateLimitError:
        logger.warning("[LLM] Groq rate limit hit (429) — serving rate limit response")
        return RATE_LIMIT_RESPONSE
 
    except Exception as exc:
        logger.error("[LLM] Groq call failed", exc_info=True)
        return FALLBACK_RESPONSE