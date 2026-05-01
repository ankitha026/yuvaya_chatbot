"""
test_chatbot.py – End-to-end test suite for the Collagreens Chatbot system.

Tests the full RAG pipeline locally WITHOUT Docker or a running server:
  - Knowledge base loading & validation
  - Query rewriting
  - Keyword retrieval
  - Semantic (embedding) retrieval
  - Hybrid retrieval + context building
  - LLM response generation (live Groq call via .env)
  - Fallback handling
  - Error/edge-case handling
  - Conversation memory (graceful degradation without Redis)
  - Full chatbot engine end-to-end

Usage:
    cd <project_root>
    python test_chatbot.py

Requirements:
    pip install -r requirements.txt
    A valid .env file with at least GROQ_API_KEY set (or another provider).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from typing import Any, Callable, List, Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# WINDOWS DLL / TORCH FIX  (must run before ANY other import)
#
# Problem:  sentence-transformers pulls in PyTorch, which on some Windows
#           machines fails with:
#             [WinError 1114] A dynamic link library (DLL) initialization
#             routine failed. Error loading …\torch\lib\c10.dll
#
# Root causes & mitigations applied here:
#   1. Force CPU-only torch so CUDA DLLs are never loaded.
#   2. Set TOKENIZERS_PARALLELISM=false to silence HuggingFace fork warnings
#      that can mask the real error.
#   3. Pre-import torch inside a try/except so a DLL failure prints a clear
#      human-readable message and exits cleanly instead of crashing config.py
#      with a cryptic traceback.
#   4. If torch loads but CUDA triggers the DLL error, restrict torch to CPU
#      via the environment variable before sentence-transformers touches it.
# ─────────────────────────────────────────────────────────────────────────────

# (a) Tell PyTorch / HuggingFace to stay on CPU – prevents CUDA DLL loading
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # hide all GPUs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # no forked tokenizer
os.environ.setdefault("TORCH_DEVICE", "cpu")               # hint for some libs

# (b) Early torch smoke-test – surface the DLL problem with a clean message
try:
    import torch  # noqa: F401  (imported for its side-effect: loading DLLs now)
    # Force CPU context so subsequent sentence-transformer calls never touch CUDA
    torch.set_default_device("cpu") if hasattr(torch, "set_default_device") else None
except OSError as _torch_err:
    _msg = str(_torch_err)
    print(
        "\n[SETUP ERROR] PyTorch failed to load its native DLLs.\n"
        f"  Detail : {_msg}\n\n"
        "  Most likely causes on Windows:\n"
        "   • The venv was created with a GPU (CUDA) version of torch but no\n"
        "     compatible GPU / CUDA runtime is present.\n"
        "   • Visual C++ Redistributable (vc_redist) is outdated or missing.\n\n"
        "  Quick fix – reinstall CPU-only torch inside your venv:\n"
        "    pip uninstall torch torchvision torchaudio -y\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "\n  Re-run test_chatbot.py after the reinstall."
    )
    sys.exit(1)
except ImportError:
    pass  # torch not installed at all – sentence-transformers will handle it

# ─────────────────────────────────────────────────────────────────────────────
# Path setup – make src importable from any working directory
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─────────────────────────────────────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg: str)  -> str: return f"{GREEN}✔  PASS{RESET}  {msg}"
def _fail(msg: str) -> str: return f"{RED}✘  FAIL{RESET}  {msg}"
def _warn(msg: str) -> str: return f"{YELLOW}⚠  WARN{RESET}  {msg}"
def _head(msg: str) -> str: return f"\n{BOLD}{CYAN}{'─'*60}\n  {msg}\n{'─'*60}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Test runner utility
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.passed: int = 0
        self.failed: int = 0
        self.warnings: int = 0
        self._failures: List[str] = []

    def record(self, name: str, success: bool, detail: str = "", warning: bool = False):
        if warning:
            print(f"  {_warn(name)}")
            if detail:
                print(f"      {YELLOW}{detail}{RESET}")
            self.warnings += 1
        elif success:
            print(f"  {_ok(name)}")
            if detail:
                print(f"      {detail}")
            self.passed += 1
        else:
            print(f"  {_fail(name)}")
            if detail:
                print(f"      {RED}{detail}{RESET}")
            self.failed += 1
            self._failures.append(f"{name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{BOLD}{'='*60}")
        print(f"  RESULTS:  {self.passed}/{total} passed  |  "
              f"{self.failed} failed  |  {self.warnings} warnings")
        print(f"{'='*60}{RESET}")
        if self._failures:
            print(f"\n{RED}{BOLD}Failed tests:{RESET}")
            for f in self._failures:
                print(f"  {RED}• {f}{RESET}")
        return self.failed == 0


results = TestResult()


def run_test(name: str, fn: Callable, *args, warning_on_fail: bool = False, **kwargs) -> Any:
    """Execute a single test function and record its outcome."""
    try:
        value = fn(*args, **kwargs)
        results.record(name, True)
        return value
    except AssertionError as e:
        results.record(name, not warning_on_fail, str(e), warning=warning_on_fail)
        return None
    except Exception as e:
        results.record(name, not warning_on_fail,
                       f"{type(e).__name__}: {e}", warning=warning_on_fail)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE BASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_base() -> Optional[list]:
    print(_head("1. Knowledge Base"))

    from src.data.knowledge_base import get_all_chunks, init_knowledge_base

    # 1a. Init without error
    def _init():
        init_knowledge_base()

    run_test("KB initializes without error", _init)

    # 1b. Returns non-empty list
    def _load():
        chunks = get_all_chunks()
        assert isinstance(chunks, list), "Expected list"
        assert len(chunks) > 0, "Knowledge base is empty"
        return chunks

    chunks = run_test("KB returns non-empty chunk list", _load)

    if chunks is None:
        return None

    # 1c. All required fields present
    def _fields():
        for c in chunks:
            assert c.id,       f"Chunk missing id: {c}"
            assert c.category, f"Chunk missing category: {c.id}"
            assert c.text,     f"Chunk missing text: {c.id}"
            assert isinstance(c.keywords, list), f"keywords not list: {c.id}"
            assert len(c.text.strip()) >= 20,    f"Text too short: {c.id}"

    run_test("All chunks have required fields (id, category, text, keywords)", _fields)

    # 1d. No duplicate IDs
    def _unique_ids():
        ids = [c.id for c in chunks]
        dupes = [i for i in ids if ids.count(i) > 1]
        assert not dupes, f"Duplicate IDs found: {set(dupes)}"

    run_test("No duplicate chunk IDs", _unique_ids)

    # 1e. Print summary
    categories = {c.category for c in chunks}
    print(f"      → Loaded {len(chunks)} chunks across {len(categories)} categories: "
          f"{', '.join(sorted(categories))}")

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUERY REWRITER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_query_rewriter():
    print(_head("2. Query Rewriter"))

    from src.utils.query_rewriter import rewrite_query

    # 2a. Contraction expansion
    def _contractions():
        r = rewrite_query("What's Collagreens?")
        assert "what is" in r, f"Expected 'what is' in '{r}'"

    run_test("Contraction expansion: what's → what is", _contractions)

    # 2b. Lowercasing
    def _lower():
        r = rewrite_query("COLLAGREENS BENEFITS")
        assert r == r.lower(), f"Result not lowercase: '{r}'"

    run_test("Query is lowercased", _lower)

    # 2c. Trailing punctuation stripped
    def _punct():
        r = rewrite_query("What is this???")
        assert not r.endswith("?"), f"Trailing punctuation not removed: '{r}'"

    run_test("Trailing punctuation removed", _punct)

    # 2d. Empty / whitespace input
    def _empty():
        r = rewrite_query("   ")
        assert isinstance(r, str), "Expected string for whitespace input"

    run_test("Empty/whitespace input returns string (not exception)", _empty)

    # 2e. Filler phrase removal
    def _filler():
        r = rewrite_query("Can you tell me about collagen?")
        assert "can you" not in r, f"Filler 'can you' not removed: '{r}'"

    run_test("Filler phrases removed (can you, please, etc.)", _filler)

    # 2f. Print some rewrites for visual inspection
    samples = [
        "What's the dosage for Collagreens?",
        "Please tell me about collagen benefits",
        "How's it different from other supplements?",
    ]
    print(f"      {'Original':<50} → Rewritten")
    for s in samples:
        print(f"      {s:<50} → {rewrite_query(s)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. KEYWORD RETRIEVER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_keyword_retriever():
    print(_head("3. Keyword Retriever"))

    from src.retrieval.keyword_retriever import keyword_retrieve

    # 3a. Basic retrieval returns list
    def _basic():
        r = keyword_retrieve("collagen skin benefits", top_k=5)
        assert isinstance(r, list), "Expected list"
        assert len(r) > 0, "No results for 'collagen skin benefits'"
        return r

    results_kw = run_test("Retrieves results for 'collagen skin benefits'", _basic)

    # 3b. Scores in [0, 1]
    def _scores():
        r = keyword_retrieve("collagen", top_k=10)
        for chunk, score in r:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {chunk.id}"

    run_test("All keyword scores in valid range [0, 1]", _scores)

    # 3c. Results sorted descending
    def _sorted():
        r = keyword_retrieve("skin hair nails collagen", top_k=10)
        scores = [s for _, s in r]
        assert scores == sorted(scores, reverse=True), "Results not sorted descending"

    run_test("Results sorted by score (descending)", _sorted)

    # 3d. top_k respected
    def _topk():
        r = keyword_retrieve("collagen", top_k=3)
        assert len(r) <= 3, f"Got {len(r)} results, expected ≤3"

    run_test("top_k parameter respected", _topk)

    # 3e. Empty query handled gracefully
    def _empty():
        r = keyword_retrieve("")
        assert isinstance(r, list), "Expected list for empty query"

    run_test("Empty query handled gracefully (no exception)", _empty)

    # 3f. Print top result
    if results_kw:
        top_chunk, top_score = results_kw[0]
        print(f"      → Top result: [{top_chunk.id}] score={top_score:.4f} "
              f"text='{top_chunk.text[:80]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEMANTIC (EMBEDDING) RETRIEVER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_embedding_retriever():
    print(_head("4. Semantic (Embedding) Retriever"))

    from src.retrieval.embedding_retriever import semantic_retrieve, ensure_index, warmup_model

    # 4a. Model warms up
    def _warmup():
        warmup_model()  # Should not raise

    run_test("Embedding model warms up without error", _warmup)

    # 4b. Index builds successfully
    def _index():
        ensure_index()

    run_test("FAISS index builds/loads without error", _index)

    # 4c. Returns relevant results
    def _retrieve():
        r = semantic_retrieve("collagen benefits for skin", top_k=5)
        assert isinstance(r, list), "Expected list"
        assert len(r) > 0, "No semantic results for collagen benefits query"
        return r

    sem_results = run_test("Semantic retrieval returns results", _retrieve)

    # 4d. Scores in valid range
    def _scores():
        r = semantic_retrieve("skin health wellness", top_k=5)
        for chunk, score in r:
            assert 0.0 <= score <= 1.001, f"Score {score:.4f} out of expected range"

    run_test("Semantic scores in valid range [0, 1]", _scores)

    # 4e. top_k limit enforced
    def _topk():
        r = semantic_retrieve("collagen", top_k=2)
        assert len(r) <= 2, f"Got {len(r)} results, expected ≤2"

    run_test("top_k limit enforced for semantic retrieval", _topk)

    # 4f. Empty query handled
    def _empty():
        r = semantic_retrieve("")
        assert isinstance(r, list)

    run_test("Empty query handled gracefully in semantic retriever", _empty)

    # Print top result
    if sem_results:
        top_chunk, top_score = sem_results[0]
        print(f"      → Top semantic result: [{top_chunk.id}] score={top_score:.4f} "
              f"text='{top_chunk.text[:80]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID RETRIEVER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_hybrid_retriever():
    print(_head("5. Hybrid Retriever"))

    from src.retrieval.hybrid_retriever import hybrid_retrieve, build_context

    test_queries = [
        ("Product overview query",     "what is collagreens product overview"),
        ("Ingredient query",           "marine collagen peptides ingredients"),
        ("Benefits query",             "skin hair nails gut wellness benefits"),
        ("Purchase/pricing query",     "where to buy collagreens price"),
        ("Off-topic irrelevant query", "pizza volcano alien spaceship robot"),
    ]

    for label, query in test_queries:
        def _run(q=query):
            r = hybrid_retrieve(q, top_k=5, similarity_threshold=0.05)
            assert isinstance(r, list), "Expected list"
            return r

        r = run_test(f"Hybrid retrieve: '{label}'", _run)

        if r is not None:
            top_score = r[0][1] if r else 0.0
            print(f"      → query='{query[:55]}' | hits={len(r)} | "
                  f"top_score={top_score:.4f}")

    # No duplicate IDs
    def _no_dupes():
        r = hybrid_retrieve("skin hair nails collagen", top_k=10)
        ids = [c.id for c, _ in r]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {[i for i in ids if ids.count(i)>1]}"

    run_test("No duplicate chunks in hybrid results", _no_dupes)

    # Sorted descending
    def _sorted():
        r = hybrid_retrieve("collagen", top_k=10)
        scores = [s for _, s in r]
        assert scores == sorted(scores, reverse=True), "Not sorted descending"

    run_test("Hybrid results sorted by score (descending)", _sorted)

    # build_context returns non-empty string for valid results
    def _context():
        r = hybrid_retrieve("what is collagreens", top_k=5)
        assert r, "No results to build context from"
        ctx = build_context(r)
        assert isinstance(ctx, str), "Context must be a string"
        assert len(ctx.strip()) > 0, "Context is empty"
        return ctx

    ctx = run_test("build_context produces non-empty string", _context)
    if ctx:
        print(f"      → Context preview: '{ctx[:120].strip()}...'")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONVERSATION MEMORY TESTS (graceful without Redis)
# ─────────────────────────────────────────────────────────────────────────────

def test_memory():
    print(_head("6. Conversation Memory"))

    # Patch Redis to simulate unavailability (no Redis needed to run locally)
    mock_redis = MagicMock()
    mock_redis.ping.side_effect = ConnectionError("Redis not available in test environment")

    # Re-instantiate memory with mocked Redis
    with patch("redis.Redis", return_value=mock_redis):
        from src.memory.memory import ConversationMemory
        mem = ConversationMemory(window=3)

    # 6a. Memory gracefully handles Redis unavailability
    def _no_redis():
        # All operations should silently degrade, not raise
        session = "test_session_" + uuid.uuid4().hex[:8]
        mem.add_user(session, "Hello Collagreens!")
        mem.add_assistant(session, "Hi! How can I help?")
        history = mem.get_history(session)
        assert isinstance(history, list), "Expected list"
        # Without Redis, returns empty list – that's acceptable graceful degradation
        return history

    history = run_test("Memory degrades gracefully without Redis (no exception)", _no_redis)

    # 6b. build_messages_for_llm always returns at least the current query
    def _build_msgs():
        session = "build_" + uuid.uuid4().hex[:8]
        msgs = mem.build_messages_for_llm(session, "What is collagreens?")
        assert isinstance(msgs, list), "Expected list"
        assert len(msgs) >= 1, "Should have at least the current message"
        assert msgs[-1]["role"] == "user", "Last message should be user"
        assert msgs[-1]["content"] == "What is collagreens?", "Content mismatch"

    run_test("build_messages_for_llm always includes current query as last message", _build_msgs)

    # 6c. enrich_query_with_context falls back gracefully
    def _enrich():
        session = "enrich_" + uuid.uuid4().hex[:8]
        enriched = mem.enrich_query_with_context(session, "what else?")
        assert isinstance(enriched, str), "Expected string"
        assert len(enriched) > 0, "Enriched query is empty"

    run_test("enrich_query_with_context handles empty history gracefully", _enrich)

    # 6d. clear does not raise
    def _clear():
        mem.clear("nonexistent_session")

    run_test("clear() on nonexistent session does not raise", _clear)

    print(f"      → Note: Full Redis-backed memory requires a running Redis instance.")

    # --- Test with real Redis if available ---
    try:
        import redis
        real_redis = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        real_redis.ping()
        _test_real_memory()
    except Exception:
        print(f"      {YELLOW}⚠  Redis not available – skipping live memory tests{RESET}")


def _test_real_memory():
    """Run additional tests when Redis is actually available."""
    from src.memory.memory import ConversationMemory

    mem = ConversationMemory(window=3)
    session = "live_test_" + uuid.uuid4().hex[:8]

    def _add_retrieve():
        mem.add_user(session, "Tell me about Collagreens")
        mem.add_assistant(session, "Collagreens is a wellness supplement.")
        history = mem.get_history(session)
        assert len(history) == 2, f"Expected 2 messages, got {len(history)}"

    run_test("[Redis LIVE] Add and retrieve messages", _add_retrieve)

    def _window_trim():
        # Add 10 turns; window=3 → max 6 messages stored
        for i in range(10):
            mem.add_user(session, f"message {i}")
            mem.add_assistant(session, f"reply {i}")
        history = mem.get_history(session)
        assert len(history) <= 6, f"Expected ≤6 messages, got {len(history)}"

    run_test("[Redis LIVE] Window trimming enforced (max 2×window messages)", _window_trim)

    def _clear_live():
        mem.clear(session)
        assert mem.get_history(session) == [], "History not cleared"

    run_test("[Redis LIVE] Session cleared successfully", _clear_live)


# ─────────────────────────────────────────────────────────────────────────────
# 7. LLM INTERFACE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_interface():
    print(_head("7. LLM Interface"))

    from src.llm.llm_interface import generate_response, FALLBACK_RESPONSE

    # 7a. Returns fallback when context is empty
    def _no_context():
        result = generate_response(messages=[], context="")
        assert result == FALLBACK_RESPONSE, f"Expected fallback, got: '{result[:80]}'"

    run_test("Returns FALLBACK_RESPONSE when context is empty", _no_context)

    # 7b. Returns fallback when context is whitespace
    def _whitespace_ctx():
        result = generate_response(
            messages=[{"role": "user", "content": "test"}],
            context="   ",
        )
        assert result == FALLBACK_RESPONSE, f"Expected fallback for whitespace context"

    run_test("Returns FALLBACK_RESPONSE for whitespace-only context", _whitespace_ctx)

    # 7c. FALLBACK_RESPONSE contains expected contact information
    def _fallback_content():
        assert "support@yuvaya.in" in FALLBACK_RESPONSE or "yuvaya.in" in FALLBACK_RESPONSE, \
            "FALLBACK_RESPONSE missing contact info"

    run_test("FALLBACK_RESPONSE contains contact info", _fallback_content)

    # 7d. Live LLM call test (real API call with Groq/whichever provider is configured)
    print(f"\n      {CYAN}Running live LLM call (requires valid API key in .env)...{RESET}")

    from src.retrieval.hybrid_retriever import hybrid_retrieve, build_context

    live_query = "What is Collagreens and what are its main benefits?"
    chunks = hybrid_retrieve(live_query, top_k=3, similarity_threshold=0.05)
    context = build_context(chunks) if chunks else ""

    def _live_call():
        assert context.strip(), "Could not build context for live LLM test"
        messages = [{"role": "user", "content": live_query}]
        t0 = time.perf_counter()
        response = generate_response(messages=messages, context=context)
        elapsed = round(time.perf_counter() - t0, 2)
        assert isinstance(response, str), "LLM response must be a string"
        assert len(response.strip()) > 10, f"LLM response too short: '{response}'"
        print(f"\n      {CYAN}Query :{RESET} {live_query}")
        print(f"      {CYAN}LLM   :{RESET} {response[:300]}...")
        print(f"      {CYAN}Time  :{RESET} {elapsed}s")
        return response

    run_test("Live LLM API call returns meaningful response", _live_call, warning_on_fail=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL CHATBOT ENGINE END-TO-END TESTS
# ─────────────────────────────────────────────────────────────────────────────

# The 5 realistic test queries (product-domain focused)
E2E_QUERIES = [
    (
        "product_overview",
        "What is Collagreens and what makes it different from other collagen supplements?",
    ),
    (
        "ingredients",
        "What type of collagen is used in Collagreens and where is it sourced from?",
    ),
    (
        "benefits_usage",
        "How long does it take to see results from Collagreens? When should I take it?",
    ),
    (
        "purchase_shipping",
        "Where can I buy Collagreens and how long does shipping take?",
    ),
    (
        "off_topic_fallback",
        "What is the best recipe for chocolate cake?",  # should trigger fallback or redirect
    ),
]


def test_chatbot_engine_mocked():
    """Test chatbot engine logic with a mocked LLM (no API calls needed)."""
    print(_head("8. Chatbot Engine (Mocked LLM)"))

    from src.chatbot import ChatbotEngine
    from src.llm.llm_interface import FALLBACK_RESPONSE

    with patch("src.llm.llm_interface.generate_response",
               return_value="[MOCKED] Collagreens is a premium wellness supplement."):

        engine = ChatbotEngine(top_k=5, similarity_threshold=0.05)

        # 8a. Engine instantiates
        run_test("ChatbotEngine instantiates without error", lambda: None)

        # 8b. Basic chat returns string
        def _basic_chat():
            r = engine.chat("What is Collagreens?", session_id="mock_s1")
            assert isinstance(r, str), "Response must be a string"
            assert len(r.strip()) > 0, "Response is empty"
            return r

        run_test("engine.chat() returns non-empty string response", _basic_chat)

        # 8c. Invalid top_k raises ValueError
        def _bad_topk():
            try:
                ChatbotEngine(top_k=0)
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass  # expected

        run_test("Invalid top_k=0 raises ValueError", _bad_topk)

        # 8d. Invalid threshold raises ValueError
        def _bad_threshold():
            try:
                ChatbotEngine(similarity_threshold=2.0)
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass

        run_test("Invalid similarity_threshold=2.0 raises ValueError", _bad_threshold)

        # 8e. Non-string query raises TypeError
        def _non_string():
            try:
                engine.chat(12345, session_id="s_err")
                raise AssertionError("Should have raised TypeError")
            except TypeError:
                pass

        run_test("Non-string query raises TypeError", _non_string)

        # 8f. Empty query raises ValueError
        def _empty_query():
            try:
                engine.chat("   ", session_id="s_empty")
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass

        run_test("Empty/whitespace query raises ValueError", _empty_query)

        # 8g. Oversized query raises ValueError
        def _oversize():
            try:
                engine.chat("x" * 2001, session_id="s_big")
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass

        run_test("Query > 2000 chars raises ValueError", _oversize)

    # 8h. Fallback triggered when retriever returns nothing
    def _fallback():
        with patch("src.chatbot.hybrid_retrieve", return_value=[]):
            from src.chatbot import ChatbotEngine
            from src.llm.llm_interface import FALLBACK_RESPONSE
            eng = ChatbotEngine()
            r = eng.chat("pizza volcano alien", session_id="s_fallback")
            assert r == FALLBACK_RESPONSE, f"Expected FALLBACK, got: '{r[:80]}'"

    run_test("FALLBACK_RESPONSE returned when retriever finds nothing", _fallback)


def test_chatbot_engine_live():
    """Run full end-to-end chatbot queries with real LLM API calls."""
    print(_head("9. Chatbot Engine – Live End-to-End Queries"))

    from src.chatbot import ChatbotEngine
    from src.llm.llm_interface import FALLBACK_RESPONSE

    # Use a fresh engine with slightly relaxed threshold for testing
    engine = ChatbotEngine(top_k=5, similarity_threshold=0.05)

    print(f"\n  {'#':<4} {'Category':<22} {'Status':<10} {'Latency':>8}  Preview")
    print(f"  {'─'*80}")

    session_id = "e2e_" + uuid.uuid4().hex[:8]

    for idx, (category, query) in enumerate(E2E_QUERIES, start=1):
        t0 = time.perf_counter()
        try:
            response = engine.chat(query, session_id=session_id)
            elapsed = round(time.perf_counter() - t0, 2)

            is_fallback = (response == FALLBACK_RESPONSE)
            status_icon = (f"{YELLOW}FALLBACK{RESET}" if is_fallback
                           else f"{GREEN}ANSWER{RESET}")

            # For the off-topic query, fallback is the EXPECTED behavior
            expected_fallback = (category == "off_topic_fallback")
            test_passed = is_fallback if expected_fallback else not is_fallback

            preview = response.replace("\n", " ")[:70]

            print(f"  {idx:<4} {category:<22} {status_icon:<10} {elapsed:>6.2f}s  {preview}")
            print(f"       {CYAN}Q:{RESET} {query}")
            print(f"       {CYAN}A:{RESET} {response[:200].replace(chr(10), ' ')}")
            print()

            results.record(
                f"E2E [{category}]",
                success=test_passed,
                detail=(f"latency={elapsed}s | "
                        + ("fallback as expected" if expected_fallback and is_fallback
                           else f"response len={len(response)}")),
            )

        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  {idx:<4} {category:<22} {RED}ERROR{RESET}    {elapsed:>6.2f}s  "
                  f"{type(e).__name__}: {str(e)[:60]}")
            results.record(f"E2E [{category}]", False,
                           f"{type(e).__name__}: {e}")


def test_chatbot_multi_turn():
    """Test multi-turn conversation continuity (follow-up queries)."""
    print(_head("10. Multi-Turn Conversation Test"))

    # Only run if Redis is available; otherwise test the fallback path
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        redis_available = True
    except Exception:
        redis_available = False

    if not redis_available:
        print(f"  {YELLOW}⚠  Redis unavailable – testing stateless fallback path{RESET}")

    from src.chatbot import ChatbotEngine

    engine = ChatbotEngine(top_k=5, similarity_threshold=0.05)
    session = "multi_turn_" + uuid.uuid4().hex[:8]

    turns = [
        "What is Collagreens?",
        "What are its main ingredients?",
        "How do I take it?",
    ]

    previous_response = None
    all_passed = True

    for i, query in enumerate(turns, start=1):
        def _turn(q=query, prev=previous_response, turn=i):
            t0 = time.perf_counter()
            response = engine.chat(q, session_id=session)
            elapsed = round(time.perf_counter() - t0, 2)
            assert isinstance(response, str), f"Turn {turn}: response must be string"
            assert len(response.strip()) > 5, f"Turn {turn}: response too short"
            print(f"      Turn {turn}: Q='{q[:60]}'")
            print(f"              A='{response[:100].replace(chr(10),' ')}...' ({elapsed}s)")
            return response

        resp = run_test(f"Multi-turn: Turn {i} responds successfully", _turn,
                        warning_on_fail=False)
        if resp:
            previous_response = resp
        else:
            all_passed = False


# ─────────────────────────────────────────────────────────────────────────────
# 11. ERROR HANDLING AND EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_error_handling():
    print(_head("11. Error Handling & Edge Cases"))

    from src.retrieval.keyword_retriever import keyword_retrieve
    from src.retrieval.hybrid_retriever import hybrid_retrieve, build_context
    from src.utils.query_rewriter import rewrite_query
    from src.llm.llm_interface import generate_response, FALLBACK_RESPONSE

    # 11a. build_context with empty list → empty string
    def _empty_context():
        ctx = build_context([])
        assert ctx == "", f"Expected empty string, got '{ctx}'"

    run_test("build_context([]) returns empty string", _empty_context)

    # 11b. hybrid_retrieve with zero-length query gracefully returns list
    def _empty_hybrid():
        # Patching internals so rewrite_query doesn't raise
        r = hybrid_retrieve("   ", top_k=5)
        assert isinstance(r, list)

    run_test("hybrid_retrieve with whitespace query returns list (no crash)", _empty_hybrid)

    # 11c. generate_response with no messages, good context → falls back gracefully
    def _empty_messages():
        ctx = "Collagreens is a wellness supplement with marine collagen."
        # Empty messages list – the function should handle this
        try:
            result = generate_response(messages=[], context=ctx)
            # May raise or return FALLBACK – either is acceptable
            assert isinstance(result, str)
        except Exception:
            pass  # Raising is also acceptable behavior

    run_test("generate_response with empty messages handles gracefully", _empty_messages)

    # 11d. Query with special characters handled
    def _special_chars():
        r = rewrite_query("What about <Collagreens> & its \"benefits\"?!")
        assert isinstance(r, str)

    run_test("rewrite_query handles special characters without crashing", _special_chars)

    # 11e. Very long query truncated safely
    def _long_query():
        long_q = "collagen " * 200   # 1800+ chars
        r = rewrite_query(long_q)
        assert isinstance(r, str)
        assert len(r) <= 1000, f"Expected truncation at 1000 chars, got {len(r)}"

    run_test("Very long query is truncated safely by rewrite_query", _long_query)

    # 11f. Numeric-only query returns list
    def _numeric():
        r = keyword_retrieve("12345 6789 00000")
        assert isinstance(r, list)

    run_test("Numeric-only query handled gracefully by keyword retriever", _numeric)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  Collagreens Chatbot – End-to-End Test Suite")
    print(f"  (No Docker required – runs locally against .env config)")
    print(f"{'='*60}{RESET}")

    # Print active LLM provider
    try:
        from src.config import settings
        print(f"\n  {CYAN}LLM Provider :{RESET} {settings.LLM_PROVIDER.upper()}")
        print(f"  {CYAN}Embedding    :{RESET} {settings.EMBEDDING_MODEL_NAME}")
        print(f"  {CYAN}Top-K        :{RESET} {settings.TOP_K_CHUNKS}")
        print(f"  {CYAN}Threshold    :{RESET} {settings.SIMILARITY_THRESHOLD}")
    except OSError as e:
        # Windows DLL failure that slipped past the early torch check
        print(f"\n  {RED}[DLL/OS ERROR] A native library failed to load:{RESET}")
        print(f"  {RED}{e}{RESET}\n")
        print(f"  {YELLOW}This is almost always a CPU-vs-CUDA torch mismatch on Windows.{RESET}")
        print(f"  {YELLOW}Fix — run these commands inside your venv:{RESET}")
        print("    pip uninstall torch torchvision torchaudio -y")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)
    except RuntimeError as e:
        # Missing API key or bad .env value caught by Settings.__init__
        print(f"\n  {RED}Config error (likely missing API key): {e}{RESET}")
        print(f"  {YELLOW}Check your .env — set the key for your chosen LLM_PROVIDER.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  {RED}Unexpected config load error: {type(e).__name__}: {e}{RESET}")
        print(f"  {YELLOW}Ensure .env is present and has a valid API key.{RESET}")
        sys.exit(1)

    total_start = time.perf_counter()

    # ── Run all test stages ──────────────────────────────────────────────────
    test_knowledge_base()
    test_query_rewriter()
    test_keyword_retriever()
    test_embedding_retriever()
    test_hybrid_retriever()
    test_memory()
    test_llm_interface()
    test_chatbot_engine_mocked()
    test_chatbot_engine_live()
    test_chatbot_multi_turn()
    test_error_handling()

    total_elapsed = round(time.perf_counter() - total_start, 2)
    print(f"\n  {CYAN}Total test time: {total_elapsed}s{RESET}")

    passed = results.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()