#  Collagreens Chatbot

 **Hybrid RAG + LLM chatbot** built for Yuvaya's *Collagreens* wellness supplement — answering customer questions accurately, with full conversation memory and multi-provider LLM support.


## 🔍 Overview

Collagreens Chatbot is a domain-specific customer support assistant for the Collagreens supplement brand. It uses a **hybrid retrieval-augmented generation (RAG)** pipeline to answer questions strictly from a curated product knowledge base — no hallucination, no off-topic drift.

**Key behaviours:**
- Answers only from product knowledge; declines off-topic queries gracefully
- Remembers conversation context across turns (Redis-backed session memory)
- Falls back to a human support contact when confidence is low

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│                 FastAPI Server                  │
│                   main.py                       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              ChatbotEngine                      │
│               chatbot.py                        │
│                                                 │
│  1. Enrich query (Redis conversation memory)    │
│  2. Rewrite / normalise query                   │
│  3. Hybrid retrieval                            │
│  4. Confidence check → fallback if weak         │
│  5. Build context string                        │
│  6. Call LLM with history + context             │
│  7. Store reply in memory                       │
└──────┬──────────────┬─────────────────┬─────────┘
       │              │                 │
       ▼              ▼                 ▼
┌────────────┐ ┌────────────┐ ┌───────────────────┐
│  Keyword   │ │  Semantic  │ │   LLM Interface   │
│ Retriever  │ │ Retriever  │ │  llm_interface.py │
│            |  │(FAISS+      │                   │
│            │ │ SentenceT.)│ │                   │                 
└────────────┘ └────────────┘ │   Groq            │
       │              │       └───────────────────┘
       └──────┬───────┘
              ▼
     ┌─────────────────┐
     │ Hybrid Retriever│  (weighted score fusion)
     │hybrid_retriever │
     └─────────────────┘
              │
              ▼
     ┌─────────────────┐
     │ Knowledge Base  │  (JSON chunks, FAISS index)
     └─────────────────┘
```

## 📁 Project Structure

```
collagreens_chatbot/
├── main.py                        # FastAPI app entry point
├── requirements.txt
├── .env.example                   # Environment variable template
│
├── src/
│   ├── config.py                  # Typed settings loaded from .env
│   ├── chatbot.py                 # Core orchestration engine
│   │
│   ├── data/
│   │   ├── knowledge_base.py      # JSON loader & KnowledgeChunk dataclass
│   │   └── collagreens_data.json  # Curated product knowledge chunks
│   │
│   ├── retrieval/
│   │   ├── keyword_retriever.py   # Stemmed token + exact-phrase scoring
│   │   ├── embedding_retriever.py # FAISS semantic retrieval
│   │   └── hybrid_retriever.py    # Score fusion + context builder
│   │
│   ├── llm/
│   │   └── llm_interface.py       # Provider-agnostic LLM adapter
│   │
│   ├── memory/
│   │   └── memory.py              # Redis-backed ConversationMemory
│   │
│   └── utils/
│       ├── query_rewriter.py      # Query normalisation pipeline
│       └── logger.py              # Structured logging setup
│
├── scripts/
│   └── preprocess_data.py         # Raw FAQ → knowledge-base JSON
│
├── cache/                         # Auto-generated FAISS index & chunk cache
├── logs/                          # Auto-generated log files
│
└── tests/
    └── test_chatbot.py            # End-to-end test suite (no Docker needed)
```

---

## 🔧 Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| Redis | 6+ (optional — memory degrades gracefully without it) |
| API key | For your chosen LLM provider |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/collagreens-chatbot.git
cd collagreens-chatbot
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)

```python
python -c "import nltk; nltk.download('punkt')"
```


## ⚙️ Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

### `.env` reference

```env
# ── LLM Provider ──────────────────────────────────────────
 groq
LLM_PROVIDER=groq

# ── API Keys (only the key for your active provider is required) ──

GROQ_API_KEY=gsk_...


GROQ_MODEL=llama-3.1-8b-instant

# ── Retrieval tuning ───────────────────────────────────────
TOP_K_CHUNKS=3              # Max chunks passed to the LLM
SIMILARITY_THRESHOLD=0.30   # Min hybrid score to include a chunk
KEYWORD_WEIGHT=0.4          # Keyword signal weight (must sum to 1.0 with SEMANTIC_WEIGHT)
SEMANTIC_WEIGHT=0.6         # Semantic signal weight

# ── Conversation memory ────────────────────────────────────
MEMORY_WINDOW=3             # Number of previous turns to retain
REDIS_HOST=localhost
REDIS_PORT=6379
SESSION_TTL=1800            # Session expiry in seconds (30 min)

# ── Server ────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

---

## ▶️ Running the Application

### Local development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first startup the server will:
1. Load the knowledge base from `src/data/collagreens_data.json`
2. Warm up the `all-MiniLM-L6-v2` embedding model
3. Build (or load from cache) the FAISS vector index

### With Redis (recommended for memory)

```bash
# Start Redis (Docker)
docker run -d -p 6379:6379 redis:7

# Start the API
uvicorn main:app --reload
```

---

## 📡 API Reference

Interactive docs are available at `http://localhost:8000/docs` once the server is running.

### `GET /`

Returns API metadata.

```json
{
  "name": "Collagreens Chatbot API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "chat": "POST /chat"
}
```

---

### `GET /health`

Health check.

```json
{
  "status": "ok",
  "llm_provider": "anthropic"
}
```

---

### `POST /chat`

Main chat endpoint.

**Request body:**

```json
{
  "message": "What are the main benefits of Collagreens?",
  "session_id": "user-abc-123"
}
```

**Response:**

```json
{
  "reply": "Collagreens supports skin elasticity, hair strength, nail health, joint comfort, and gut wellness...",
  "session_id": "user-abc-123",
  "latency": 0.843
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Collagreens?", "session_id": "my-session"}'
```

**Python example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "How long before I see results?",
        "session_id": "my-session",
    },
)
print(response.json()["reply"])
```

---

## 🧪 Testing

The project ships with a self-contained end-to-end test suite that runs the full RAG pipeline **without Docker or a running server**.

```bash
# From the project root (with .env configured)
python test_chatbot.py
```
