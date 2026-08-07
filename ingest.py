"""
ingest.py — build a multi-book medical vector DB in Chroma.

Multi-book design:
  - All books live in ONE shared collection (COLLECTION below).
  - Every chunk is tagged with `source` (a short book label) so retrieval can
    search all books at once and citations can name the book each fact came from.
  - Idempotent PER SOURCE: re-running a book already ingested is skipped; a NEW
    book is added to the same collection.

To add a new book: change the 3 per-book settings, drop its .jsonl files in, re-run.

Fixes baked in: long blocks are chunked (MiniLM truncates at 256 tokens), sub-3-char
noise skipped, index + reference-list blocks dropped, list-valued metadata filtered.
"""

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ---- per-book settings: change these 3 for each new book, then re-run --------
SOURCE      = "scott_brown"                                          # short label for THIS book
BLOCKS_FILE = "9thScottBrownsOtorhinolaryngology_blocks_clean.jsonl"
TABLES_FILE = "tables.jsonl"
# -----------------------------------------------------------------------------
DB_PATH     = "chroma_db"
COLLECTION  = "medical_library"      # shared across all books; MUST match query.py
BATCH       = 500
MIN_CHARS   = 3                      # skip sub-3-char noise (page numbers, stray marks)
OVERLAP     = 32                     # tokens of overlap between adjacent chunks
DROP_INDEX  = True                   # skip the book's index pages. Flip to keep.
DROP_REFS   = True                   # skip bibliography/reference-list blocks. Flip to keep.

model      = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer  = model.tokenizer                       # MiniLM's own WordPiece tokenizer
MAX_TOKENS = model.max_seq_length - 2              # reserve [CLS]/[SEP]; = 254 here

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION)

# --- guard PER SOURCE: don't re-embed a book that's already in the library ---
if collection.get(where={"source": SOURCE}, limit=1)["ids"]:
    print(f"Source '{SOURCE}' is already in '{COLLECTION}' — skipping.")
    raise SystemExit


def scalar_meta(d):
    """Chroma metadata must be non-None scalars; drop lists/dicts/None."""
    return {k: v for k, v in d.items()
            if v is not None and isinstance(v, (str, int, float, bool))}


def is_reference(r):
    """
    True for bibliography/reference-list blocks — noise for a clinical RAG.
    Section labelled 'references'/'bibliography', OR (section is None) a dense
    numbered list with journal-year markers. Strict enough that body prose citing
    a study or two inline does NOT match.
    """
    sec = (r.get("section") or "").lower()
    if "reference" in sec or "bibliograph" in sec:
        return True
    t = r.get("text") or ""
    numbered = len(re.findall(r"(?m)^\s*\d{1,3}[.\t]", t))
    years    = len(re.findall(r"(19|20)\d{2};", t))
    return numbered >= 4 and years >= 3


def chunk_text(text):
    """Split over-budget text into overlapping token windows, each <= MAX_TOKENS,
    sliced from the ORIGINAL string (faithful citations), snapped to whitespace."""
    enc = tokenizer(text, add_special_tokens=False,
                    truncation=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n <= MAX_TOKENS:
        return [text]
    step = MAX_TOKENS - OVERLAP
    pieces = []
    for start in range(0, n, step):
        end = min(start + MAX_TOKENS, n)
        c_start = offsets[start][0]
        c_end   = offsets[end - 1][1]
        while c_start > 0 and not text[c_start - 1].isspace():
            c_start -= 1
        while c_end < len(text) and not text[c_end].isspace():
            c_end += 1
        piece = text[c_start:c_end].strip()
        if piece:
            pieces.append(piece)
        if end == n:
            break
    return pieces


ids, docs, metas = [], [], []
skipped = dropped_refs = chunked_blocks = 0

# --- blocks (prose) ---
with open(BLOCKS_FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        text = (r.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if DROP_INDEX and "index" in (r.get("section") or "").lower():
            skipped += 1
            continue
        if DROP_REFS and is_reference(r):
            dropped_refs += 1
            continue

        pieces = chunk_text(text)
        if len(pieces) > 1:
            chunked_blocks += 1
        base = f"{SOURCE}_p{r['page']}_b{r['block_index']}"
        for k, piece in enumerate(pieces):
            cid = base if len(pieces) == 1 else f"{base}_c{k}"
            ids.append(cid)
            docs.append(piece)
            metas.append(scalar_meta({
                "source": SOURCE,                 # <-- the multi-book tag
                "type": "prose",
                "doc": r.get("doc"),
                "page": r.get("page"),
                "block_index": r.get("block_index"),
                "section": r.get("section"),
                "is_heading": r.get("is_heading"),
                "chunk_index": k,
                "n_chunks": len(pieces),
            }))

# --- tables (embed the rendered markdown) ---
with open(TABLES_FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        md = (r.get("markdown") or "").strip()
        if len(md) < MIN_CHARS:
            skipped += 1
            continue
        pieces = chunk_text(md)
        base = f"{SOURCE}_p{r['page']}_t{r['table_index']}"
        for k, piece in enumerate(pieces):
            cid = base if len(pieces) == 1 else f"{base}_c{k}"
            ids.append(cid)
            docs.append(piece)
            metas.append(scalar_meta({
                "source": SOURCE,
                "type": "table",
                "doc": r.get("doc"),
                "page": r.get("page"),
                "table_index": r.get("table_index"),
                "n_rows": r.get("n_rows"),
                "n_cols": r.get("n_cols"),
                "chunk_index": k,
                "n_chunks": len(pieces),
            }))

print(f"[{SOURCE}] prepared {len(docs)} documents "
      f"({skipped} noise skipped, {dropped_refs} references dropped, "
      f"{chunked_blocks} long blocks chunked).")
print(f"Embedding + adding in batches of {BATCH}...")

for start in range(0, len(docs), BATCH):
    b_docs = docs[start:start + BATCH]
    b_emb  = model.encode(b_docs, show_progress_bar=False).tolist()
    collection.add(
        ids=ids[start:start + BATCH],
        documents=b_docs,
        embeddings=b_emb,
        metadatas=metas[start:start + BATCH],
    )
    print(f"  added {min(start + BATCH, len(docs))}/{len(docs)}")

print(f"Done. '{COLLECTION}' now holds {collection.count()} items total.")
