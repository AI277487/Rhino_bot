"""
ingest.py — build a multi-book medical vector DB in Chroma.

Multi-book: all books in ONE collection, each chunk tagged `source`.
Search-all design: no routing; citations name the book + page.

TABLE RETRIEVAL FIX (Bug B):
  A table's pipe-delimited markdown embeds terribly, so natural-language
  questions ("give me the table of X") never match it — even though it's in
  the DB. Fix: STORE the markdown (so the model still reads the real rows),
  but EMBED a caption-weighted, de-piped version so the caption (which is
  usually almost the user's question) dominates the search vector.
  This works because we pass explicit embeddings= to Chroma, so the stored
  document and the embedded text are allowed to differ.

Also baked in: long-block chunking (MiniLM truncates at 256 tokens),
sub-3-char noise skipped, index + reference-list blocks dropped.
"""

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ---- per-book settings: change these 3 for each new book, then re-run --------
SOURCE      = "scott_brown"
BLOCKS_FILE = "9thScottBrownsOtorhinolaryngology.blocks.clean.jsonl"
TABLES_FILE = "tables.jsonl"
# -----------------------------------------------------------------------------
DB_PATH     = "chroma_db"
COLLECTION  = "medical_library"      # shared across all books; MUST match query.py
BATCH       = 500
MIN_CHARS   = 3
OVERLAP     = 32
DROP_INDEX  = True 
DROP_REFS   = True

model      = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer  = model.tokenizer
MAX_TOKENS = model.max_seq_length - 2

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION)

if collection.get(where={"source": SOURCE}, limit=1)["ids"]:
    print(f"Source '{SOURCE}' is already in '{COLLECTION}' — skipping.")
    raise SystemExit


def scalar_meta(d):
    return {k: v for k, v in d.items()
            if v is not None and isinstance(v, (str, int, float, bool))}


def is_reference(r):
    sec = (r.get("section") or "").lower()
    if "reference" in sec or "bibliograph" in sec:
        return True
    t = r.get("text") or ""
    numbered = len(re.findall(r"(?m)^\s*\d{1,3}[.\t]", t))
    years    = len(re.findall(r"(19|20)\d{2};", t))
    return numbered >= 4 and years >= 3


def is_running_header(text):
    """
    True for running page-headers like '79: Granulomatous conditions of the nose 857'
    — chapter-number, colon, title, trailing page number. These repeat once per page
    across a whole chapter and, being short and topical, flood retrieval with noise
    while carrying no real content. ~1,196 of them in this book.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    return len(t) < 90 and re.match(r"^\d+:\s+.+?\s+\d+$", t) is not None


def chunk_text(text):
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


# --- table helpers (Bug B) ---------------------------------------------------

def table_caption(md):
    """Pull the human caption from a table's first markdown row, e.g.
    '| TABLE 79.1 Granulomatous conditions... |' -> 'Granulomatous conditions...'."""
    first = md.splitlines()[0] if md else ""
    cells = [c.strip() for c in first.strip().strip("|").split("|")]
    head  = next((c for c in cells if c), "")
    m = re.match(r"(?i)^table\s+[\d.]+\s+(.*)$", head)
    return (m.group(1).strip() if m else head).strip()


def flatten_table(md):
    """Pipe/dash markdown -> space-separated words, so real terms (not pipes)
    drive the embedding."""
    text = md.replace("|", " ")
    text = re.sub(r"-{2,}", " ", text)          # kill |---| separators / long dashes
    return re.sub(r"\s+", " ", text).strip()


def table_embed_text(md_piece, caption):
    """The text we EMBED for a table piece: caption weighted to the front
    (survives any truncation), then the de-piped words."""
    flat = flatten_table(md_piece)
    if caption:
        return f"Table: {caption}. {caption}. {flat}"
    return flat


# docs   = what we STORE + display (markdown for tables, prose for blocks)
# embeds = what we EMBED for search (caption-weighted for tables, == doc for prose)
ids, docs, embeds, metas = [], [], [], []
skipped = dropped_refs = chunked_blocks = 0

# --- blocks (prose): embed == stored text ---
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
        if is_running_header(text):
            skipped += 1
            continue

        pieces = chunk_text(text)
        if len(pieces) > 1:
            chunked_blocks += 1
        base = f"{SOURCE}_p{r['page']}_b{r['block_index']}"
        for k, piece in enumerate(pieces):
            cid = base if len(pieces) == 1 else f"{base}_c{k}"
            ids.append(cid)
            docs.append(piece)
            embeds.append(piece)                      # prose: embed the prose itself
            metas.append(scalar_meta({
                "source": SOURCE, "type": "prose",
                "doc": r.get("doc"), "page": r.get("page"),
                "block_index": r.get("block_index"),
                "section": r.get("section"),
                "is_heading": r.get("is_heading"),
                "chunk_index": k, "n_chunks": len(pieces),
            }))

# --- tables: STORE markdown, EMBED caption-weighted text ---
with open(TABLES_FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        md = (r.get("markdown") or "").strip()
        if len(md) < MIN_CHARS:
            skipped += 1
            continue
        caption = table_caption(md)
        pieces  = chunk_text(md)
        base = f"{SOURCE}_p{r['page']}_t{r['table_index']}"
        for k, piece in enumerate(pieces):
            cid = base if len(pieces) == 1 else f"{base}_c{k}"
            ids.append(cid)
            docs.append(piece)                        # DISPLAY: the real markdown
            embeds.append(table_embed_text(piece, caption))  # SEARCH: caption-weighted
            metas.append(scalar_meta({
                "source": SOURCE, "type": "table",
                "caption": caption,                   # handy for display/citations
                "doc": r.get("doc"), "page": r.get("page"),
                "table_index": r.get("table_index"),
                "n_rows": r.get("n_rows"), "n_cols": r.get("n_cols"),
                "chunk_index": k, "n_chunks": len(pieces),
            }))

print(f"[{SOURCE}] prepared {len(docs)} documents "
      f"({skipped} noise skipped, {dropped_refs} references dropped, "
      f"{chunked_blocks} long blocks chunked).")
print(f"Embedding + adding in batches of {BATCH}...")

for start in range(0, len(docs), BATCH):
    b_docs   = docs[start:start + BATCH]
    b_embtxt = embeds[start:start + BATCH]
    b_emb    = model.encode(b_embtxt, show_progress_bar=False).tolist()  # embed SEARCH text
    collection.add(
        ids=ids[start:start + BATCH],
        documents=b_docs,                             # store DISPLAY text
        embeddings=b_emb,
        metadatas=metas[start:start + BATCH],
    )
    print(f"  added {min(start + BATCH, len(docs))}/{len(docs)}")

print(f"Done. '{COLLECTION}' now holds {collection.count()} items total.")
