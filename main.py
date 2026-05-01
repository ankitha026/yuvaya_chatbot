"""
main.py – FastAPI application entry point for the Collagreens Chatbot.

Endpoints:
  POST /chat       – Main chat endpoint
  GET  /health     – Health check
  GET  /           – API info
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.chatbot import engine
from src.config import settings
from src.data.knowledge_base import init_knowledge_base
from src.llm.llm_interface import FALLBACK_RESPONSE
from src.retrieval.embedding_retriever import (
    ensure_index,
    warmup_model,
)
from src.utils.logger import setup_logging

# ─────────────────────────────────────────────────────────────
# Logging setup (must be first)
# ─────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# Timeout protection
CHAT_REQUEST_TIMEOUT = 30


# ─────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Collagreens Chatbot starting up ===")
    logger.info(f"LLM Provider : {settings.LLM_PROVIDER}")
    logger.info(f"Top-K Chunks : {settings.TOP_K_CHUNKS}")
    logger.info(f"Threshold    : {settings.SIMILARITY_THRESHOLD}")

    try:
        # Load knowledge base
        init_knowledge_base()

        # Warm model + build FAISS index
        warmup_model()
        ensure_index()

        logger.info("Embedding index ready. Server is live.")

    except Exception as exc:
        logger.exception(f"Startup initialization failed: {exc}")

    yield

    logger.info("=== Collagreens Chatbot shutting down ===")


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Collagreens Chatbot API",
    description="Hybrid RAG + LLM chatbot for Yuvaya's Collagreens wellness supplement.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────
# CORS
# NOTE:
# Replace "*" with frontend domain in production
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yuvaya.vercel.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query",
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session ID for conversation memory continuity. "
            "If omitted, a new session is started."
        ),
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """
        Strict request validation and sanitization.
        """
        if not value or not value.strip():
            raise ValueError("Message cannot be empty.")

        cleaned = value.strip()

        # Prevent extremely repetitive payloads
        if len(set(cleaned)) <= 2 and len(cleaned) > 20:
            raise ValueError("Invalid message format.")

        # Basic injection/prompt abuse filtering
        blocked_patterns = [
            r"<script.*?>",
            r"</script>",
            r"DROP\s+TABLE",
            r"SELECT\s+\*",
            r"--",
        ]

        for pattern in blocked_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                raise ValueError("Unsafe input detected.")

        return cleaned


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    latency: float


# ─────────────────────────────────────────────────────────────
# Middleware – request lifecycle logging
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    request_id = str(uuid.uuid4())

    logger.info(
        f"[RequestStart] "
        f"id={request_id} "
        f"method={request.method} "
        f"path={request.url.path}"
    )

    try:
        response = await call_next(request)

    except Exception as exc:
        logger.exception(
            f"[RequestFailure] id={request_id} error={type(exc).__name__}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error."
            },
        )

    latency = round(time.perf_counter() - start_time, 4)

    logger.info(
        f"[RequestEnd] "
        f"id={request_id} "
        f"status={response.status_code} "
        f"latency={latency}s"
    )

    response.headers["X-Process-Time"] = str(latency)

    return response


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["info"])
async def root():
    return {
        "name": "Collagreens Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
    }


@app.get("/health", tags=["info"])
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    - Accepts a user message and optional session_id
    - Returns assistant response and latency
    """

    start_time = time.perf_counter()

    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(
            f"[ChatRequest] "
            f"session_id={session_id} "
            f"message_length={len(request.message)}"
        )

        # Run blocking chatbot logic in threadpool
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                engine.chat,
                query=request.message,
                session_id=session_id,
            ),
            timeout=CHAT_REQUEST_TIMEOUT,
        )

    except asyncio.TimeoutError:
        logger.error(
            f"[ChatTimeout] session_id={session_id}"
        )

        reply = (
            "The request took too long to process. "
            "Please try again in a moment.\n\n"
            f"{FALLBACK_RESPONSE}"
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            f"[ChatFailure] session_id={session_id} error={exc}"
        )

        reply = FALLBACK_RESPONSE

    latency = round(time.perf_counter() - start_time, 4)

    logger.info(
        f"[ChatSuccess] "
        f"session_id={session_id} "
        f"latency={latency}s"
    )

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        latency=latency,
    )


# ─────────────────────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    logger.warning(
        f"[HTTPException] "
        f"path={request.url.path} "
        f"status={exc.status_code}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        f"[UnhandledException] "
        f"path={request.url.path} "
        f"error={type(exc).__name__}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "An unexpected error occurred. "
                "Please try again later."
            )
        },
    )