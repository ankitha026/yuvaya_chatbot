"""
scripts/preprocess_data.py
──────────────────────────
Converts raw FAQ-style text (from a .txt or .docx export) into the
structured JSON schema used by the Collagreens chatbot knowledge base.

Usage:
    python scripts/preprocess_data.py --input raw_data.txt --output src/data/collagreens_data.json
    python scripts/preprocess_data.py --input raw_data.txt          # overwrites existing data
    python scripts/preprocess_data.py --validate                    # validates existing JSON only

The script:
  1. Parses Q&A pairs from raw text (handles numbered lists, bold markers, dashes)
  2. Enriches short answers (single-word like "Yes", "No")
  3. Deduplicates similar questions (Jaccard similarity)
  4. Auto-assigns categories based on keyword matching
  5. Auto-generates keyword tags from question + answer
  6. Validates schema before writing output
  7. Logs all skipped/bad entries
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Category rules ────────────────────────────────────────────────────────────
CATEGORY_RULES = [
    ("returns_policy",   ["cancel", "return", "refund", "replacement", "exchange", "damaged", "tampered", "mismatched", "partial order"]),
    ("shipping_policy",  ["shipping", "delivery", "dispatch", "track", "awb", "cod", "cash on delivery", "next day", "same day", "charges"]),
    ("sales_purchase",   ["buy", "purchase", "price", "cost", "pack", "bundle", "subscription", "combo", "payment", "upi", "whatsapp order"]),
    ("quality_trust",    ["trust", "safe", "tested", "lab", "third party", "side effect", "kidney", "harm", "fake", "addictive", "premium", "quality"]),
    ("ingredients",      ["stevia", "moringa", "spirulina", "chlorella", "amla", "beetroot", "ginger", "curcumin", "sweetener", "ingredient", "chemical", "preservative"]),
    ("benefits",         ["skin", "hair", "nail", "joint", "gut", "digestion", "energy", "immunity", "detox", "glow", "wrinkle", "bloating"]),
    ("usage",            ["how to", "when", "dosage", "sachet", "mix", "water", "milk", "juice", "result", "week", "daily", "routine", "miss", "gym", "exercise", "diet"]),
    ("support",          ["contact", "email", "support", "hello@", "customer service", "business hours"]),
    ("product_overview", []),  # fallback
]


def assign_category(question: str, answer: str) -> str:
    combined = (question + " " + answer).lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in combined for kw in keywords):
            return category
    return "product_overview"


# ── Keyword extraction ────────────────────────────────────────────────────────
STOPWORDS = {"a","an","the","is","it","in","of","to","and","or","for","on","at","be","do","we","you","i","my","your","our","this","that","are","has","have","with","from","by","not","no","can","will","if","as","but","so","its","was","were","they","their","them","all","any","may","should","would","could","does","did","just","also","than","then","when","what","how","why","who","which","where"}

def extract_keywords(question: str, answer: str) -> list[str]:
    """Extract meaningful keyword phrases from question and answer."""
    keywords = set()
    # Add the question itself (cleaned)
    q_clean = re.sub(r"[^\w\s]", "", question.lower()).strip()
    if len(q_clean) > 5:
        keywords.add(q_clean)

    # Extract noun phrases (2-3 word windows) from question
    words = [w for w in q_clean.split() if w not in STOPWORDS and len(w) > 2]
    for i in range(len(words)):
        keywords.add(words[i])
        if i + 1 < len(words):
            keywords.add(f"{words[i]} {words[i+1]}")

    # Extract key terms from answer
    ans_words = [w for w in re.sub(r"[^\w\s]", "", answer.lower()).split()
                 if w not in STOPWORDS and len(w) > 3]
    keywords.update(ans_words[:8])

    # Filter noise
    keywords = {k for k in keywords if len(k) > 2 and not k.isdigit()}
    return sorted(keywords)[:8]


# ── Answer enrichment ─────────────────────────────────────────────────────────
SHORT_ANSWER_THRESHOLD = 60  # characters

def enrich_answer(question: str, answer: str) -> str:
    """Expand very short answers into complete sentences."""
    answer = answer.strip()
    if len(answer) >= SHORT_ANSWER_THRESHOLD:
        return answer

    q_lower = question.lower()
    a_lower = answer.lower()

    # Yes/No enrichment based on question context
    if a_lower in ("yes", "yes.", "yes,"):
        if "safe" in q_lower:
            return f"Yes, Collagreens is safe for daily use when taken as recommended (1 sachet per day)."
        if "natural" in q_lower or "natural ingredient" in q_lower:
            return f"Yes, Collagreens is made with 100% natural ingredients — no artificial additives, sweeteners, or preservatives."
        if "gluten" in q_lower:
            return "Yes, Collagreens is completely gluten-free."
        if "keto" in q_lower:
            return "Yes, Collagreens is keto-friendly due to its low sugar content and natural sweetener (stevia)."
        if "lab test" in q_lower or "tested" in q_lower:
            return "Yes, every batch of Collagreens is third-party tested for microbial contamination, heavy metals, pesticide residue, and more."
        if "vegetarian" in q_lower or "vegan" in q_lower:
            return "No, Collagreens is not suitable for vegetarians or vegans. The collagen is sourced from marine fish."
        return f"Yes — {question.lower().rstrip('?')} when it comes to Collagreens."

    if a_lower in ("no", "no.", "no,"):
        if "artificial" in q_lower or "sweetener" in q_lower:
            return "No, Collagreens contains zero artificial sweeteners. It is lightly sweetened with stevia, a natural low-calorie sweetener."
        if "preservative" in q_lower:
            return "No, Collagreens contains no harmful preservatives or artificial additives."
        if "chemical" in q_lower:
            return "No, Collagreens contains no unnecessary chemicals or synthetic additives."
        if "prescription" in q_lower:
            return "No prescription is required to purchase or use Collagreens. It is a food supplement available to all adults."
        return f"No — {question.lower().rstrip('?')} is not a concern with Collagreens."

    # Short answer — prepend question context
    if len(answer) < 30:
        return f"{answer} — regarding '{question.rstrip('?')}' and Collagreens."

    return answer


# ── Deduplication ─────────────────────────────────────────────────────────────
def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(pairs: list[dict], threshold: float = 0.75) -> list[dict]:
    """Remove near-duplicate Q&A pairs, merging keywords."""
    kept = []
    for item in pairs:
        duplicate_found = False
        for existing in kept:
            if jaccard(item["question"], existing["question"]) >= threshold:
                # Merge keywords instead of dropping
                existing["keywords"] = list(set(existing["keywords"] + item["keywords"]))[:10]
                logger.info(f"[Dedup] Merged '{item['question'][:60]}' into existing entry")
                duplicate_found = True
                break
        if not duplicate_found:
            kept.append(item)
    return kept


# ── Raw text parser ───────────────────────────────────────────────────────────
def parse_raw_text(text: str) -> list[dict]:
    """
    Parse raw FAQ text into Q&A pairs.
    Handles formats:
      - **Question?**\nAnswer
      - Q: Question\nA: Answer
      - 1. Question\nAnswer
      - - Question\nAnswer
    """
    lines = text.split("\n")
    pairs = []
    current_q = None
    answer_lines = []

    # Clean markdown bold markers
    def clean(s: str) -> str:
        s = re.sub(r"\*+", "", s)  # remove ** bold
        s = re.sub(r"^[-\d.]+\s*", "", s)  # remove leading bullets/numbers
        s = s.strip()
        return s

    for line in lines:
        line = line.strip()
        if not line:
            # Blank line signals end of an answer block
            if current_q and answer_lines:
                answer = " ".join(answer_lines).strip()
                pairs.append({"question": current_q, "answer": answer})
                current_q = None
                answer_lines = []
            continue

        cleaned = clean(line)

        # Detect question line: ends with ? OR is bold (**text**)
        is_question = (
            cleaned.endswith("?")
            or (line.startswith("**") and line.endswith("**"))
            or re.match(r"^\d+\.\s+\w", line) and "?" in line
        )

        if is_question:
            # Save previous pair
            if current_q and answer_lines:
                answer = " ".join(answer_lines).strip()
                pairs.append({"question": current_q, "answer": answer})
                answer_lines = []
            current_q = re.sub(r"\?+$", "?", cleaned)
        elif current_q:
            answer_lines.append(cleaned)

    # Flush last pair
    if current_q and answer_lines:
        pairs.append({"question": current_q, "answer": " ".join(answer_lines).strip()})

    logger.info(f"[Parser] Extracted {len(pairs)} Q&A pairs from raw text")
    return pairs


# ── Build structured chunks ───────────────────────────────────────────────────
def build_chunks(pairs: list[dict]) -> list[dict]:
    chunks = []
    for idx, pair in enumerate(pairs, start=1):
        question = pair["question"].strip()
        answer = pair["answer"].strip()

        if not question or not answer:
            logger.warning(f"[Build] Skipping empty pair at index {idx}")
            continue

        enriched_answer = enrich_answer(question, answer)
        category = assign_category(question, enriched_answer)
        keywords = extract_keywords(question, enriched_answer)

        # Build chunk text as a natural-language paragraph
        chunk_text = f"{enriched_answer}"

        chunk = {
            "id": f"chunk_{idx:04d}",
            "category": category,
            "text": chunk_text,
            "keywords": keywords,
        }
        chunks.append(chunk)

    logger.info(f"[Build] Built {len(chunks)} structured chunks")
    return chunks


# ── Schema validation ─────────────────────────────────────────────────────────
def validate_output(chunks: list[dict]) -> bool:
    errors = 0
    ids = set()
    for chunk in chunks:
        cid = chunk.get("id", "?")
        if not chunk.get("id"):
            logger.error(f"Missing id in chunk: {chunk}")
            errors += 1
        if cid in ids:
            logger.error(f"Duplicate id: {cid}")
            errors += 1
        ids.add(cid)
        if not chunk.get("category"):
            logger.error(f"Missing category in {cid}")
            errors += 1
        if not chunk.get("text") or len(chunk["text"]) < 20:
            logger.error(f"Text too short in {cid}: {chunk.get('text','')[:50]}")
            errors += 1
        if not chunk.get("keywords") or len(chunk["keywords"]) < 2:
            logger.error(f"Insufficient keywords in {cid}")
            errors += 1

    if errors:
        logger.error(f"Validation FAILED with {errors} error(s)")
        return False
    logger.info(f"Validation PASSED — {len(chunks)} chunks, {len(ids)} unique IDs")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess raw FAQ data for Collagreens chatbot")
    parser.add_argument("--input",    help="Path to raw text/docx file", default=None)
    parser.add_argument("--output",   help="Output JSON path", default="src/data/collagreens_data.json")
    parser.add_argument("--validate", action="store_true", help="Validate existing JSON only")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.validate:
        if not output_path.exists():
            logger.error(f"File not found: {output_path}")
            sys.exit(1)
        with open(output_path) as f:
            chunks = json.load(f)
        ok = validate_output(chunks)
        sys.exit(0 if ok else 1)

    if not args.input:
        logger.error("--input is required unless --validate is used")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Read input
    if input_path.suffix == ".docx":
        try:
            import subprocess
            result = subprocess.run(["extract-text", str(input_path)], capture_output=True, text=True)
            raw_text = result.stdout
        except Exception:
            logger.error("extract-text not available. Export your .docx to .txt first.")
            sys.exit(1)
    else:
        raw_text = input_path.read_text(encoding="utf-8")

    # Pipeline
    pairs   = parse_raw_text(raw_text)
    pairs   = deduplicate(pairs)
    chunks  = build_chunks(pairs)

    if not validate_output(chunks):
        logger.error("Aborting — validation failed. Fix errors above.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    print(f"\nDone! {len(chunks)} knowledge chunks written to {output_path}")
    print("Restart uvicorn to reload the new data.")


if __name__ == "__main__":
    main()
