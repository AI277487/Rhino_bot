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
sub-15-char noise skipped, index + reference-list blocks dropped.

DOCLING TABLES (cummings):
  Docling emits two caption shapes — an explicit `caption` field ("TABLE 1.3
  ...") for some tables, and null for others where the caption is baked into
  markdown row 1. table_caption() takes the explicit field when present and
  normalizes BOTH through the same stripper. A small drop filter removes the
  two non-content shapes Docling misread as tables: stray "Fig. N" lines and
  the book's video-contents lists. BOX callouts are KEPT.

RESUMABLE INGEST:
  Embedding 50k+ docs on a CPU-only laptop can exhaust RAM and segfault.
  This ingest checkpoints each batch to a progress file and can resume after
  a crash. upsert() (not add()) makes re-running an overlapping batch safe.
  The "already ingested" guard is resume-aware: it only blocks when the
  source is present AND there is no in-progress checkpoint file.
"""

import os
import json
import re
import gc
import chromadb
from is_nav_table import is_nav_table
from sentence_transformers import SentenceTransformer

# ---- per-book settings: change these 3 for each new book, then re-run --------
SOURCE      = "shaumbaugh"
BLOCKS_FILE = "shaumbaugh.blocks.jsonl"
TABLES_FILE = "shaumbaugh.tables.jsonl"
# -----------------------------------------------------------------------------
DB_PATH     = "chroma_db"
COLLECTION  = "medical_library"      # shared across all books; MUST match query.py
BATCH       = 128
MIN_CHARS   = 15
OVERLAP     = 32
DROP_INDEX  = True
DROP_REFS   = True

PROGRESS_FILE = f"ingest_progress_{SOURCE}.txt"

model      = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer  = model.tokenizer
MAX_TOKENS = model.max_seq_length - 2

client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION)

# Guard: stop ONLY if the source is already present AND we're not mid-resume.
# A progress file means a prior run crashed partway — allow it to continue.
if (collection.get(where={"source": SOURCE}, limit=1)["ids"]
        and not os.path.exists(PROGRESS_FILE)):
    print(f"Source '{SOURCE}' is already in '{COLLECTION}' — skipping.")
    print(f"(To rebuild it: delete its rows and remove {PROGRESS_FILE} first.)")
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
    while carrying no real content.
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

def table_caption(md, explicit=None):
    """Return a clean, consistent caption for a table.

    If Docling gave us an explicit caption field (e.g. 'TABLE 1.3 Examples of
    Outcomes Measures...'), use it; otherwise parse it from the first markdown
    row (the old pdfplumber / Scott-Brown convention). Either way the result is
    normalized the SAME way — a leading 'TABLE N.N' is stripped — so captions
    across every book are consistent. Non-TABLE heads (e.g. 'BOX 10.3 ...') are
    returned intact.
    """
    if explicit:
        head = explicit.strip()
    else:
        first = md.splitlines()[0] if md else ""
        cells = [c.strip() for c in first.strip().strip("|").split("|")]
        head  = next((c for c in cells if c), "")
    m = re.match(r"(?i)^table\s+[\d.]+\s+(.*)$", head)
    return (m.group(1).strip() if m else head).strip()


def is_droppable_table(md, caption):
    """True for the two non-content shapes Docling misread as tables:
       (1) stray 'Fig. N ...' captions, and
       (2) the book's video-contents lists (rows of 'NN.N  Title' with no
           TABLE/BOX marker near the top).
       BOX callouts and headerless real tables are KEPT."""
    first_line = (md.splitlines()[0] if md else "").strip()

    is_figure_caption = re.match(r"(?i)^fig\.?\s+\d", first_line) is not None

    head_cells = [c.strip() for c in first_line.strip("|").split("|") if c.strip()]
    first_cell = head_cells[0] if head_cells else ""
    is_video_list = (re.match(r"^\d+\.\d+$", first_cell) is not None
                     and "table" not in md[:200].lower()
                     and "box"   not in md[:200].lower())

    return is_figure_caption or is_video_list
def is_junk_table(md, raw_caption):
    """Drop pdfplumber misreads: section-divider pages ('SECTION 1: OTOLOGY')
    and shattered-figure fragments ('K+', 'SL K+'). Junk = NO real 'TABLE X.Y'
    caption AND almost no content. Any real captioned or content-bearing table
    is kept. Pass the RAW caption r.get('caption'), not the stripped one."""
    if raw_caption and re.match(r"(?i)^table\s+[\d.]+", raw_caption.strip()):
        return False
    first = (md.splitlines()[0] if md else "").strip().strip("|").strip()
    if re.match(r"(?i)^table\s+[\d.]+", first):
        return False
    content_chars = len(re.sub(r"[|\-\s]", "", md or ""))
    return content_chars < 60

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
dropped_tables = 0

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
                "is_figure": r.get("is_figure"),
                "chunk_index": k, "n_chunks": len(pieces),
            }))

# --- tables: STORE markdown, EMBED caption-weighted text ---
with open(TABLES_FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        md = (r.get("markdown") or "").strip().replace("\x07", "")
        if len(md) < MIN_CHARS:
            skipped += 1
            continue

        caption = table_caption(md, explicit=r.get("caption"))

        # drop only Docling's misreads (Fig. lines, video-contents lists);
        # BOX callouts and headerless real tables are kept.
        if (is_droppable_table(md, caption)
                or is_junk_table(md, r.get("caption"))
                or is_nav_table(md)):
            dropped_tables += 1
            continue

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
      f"{dropped_tables} fig/video tables dropped, "
      f"{chunked_blocks} long blocks chunked).")

# --- resumable embed + add -------------------------------------------------
# load which batches already succeeded (survives a crash/restart)
done_batches = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as pf:
        done_batches = {int(x) for x in pf.read().split() if x.strip()}
    print(f"resuming: {len(done_batches)} batches already done")

n_batches = (len(docs) + BATCH - 1) // BATCH
print(f"Embedding + adding {len(docs)} docs in {n_batches} batches of {BATCH}...")

for bi in range(n_batches):
    if bi in done_batches:
        continue
    start = bi * BATCH
    end   = min(start + BATCH, len(docs))

    b_ids    = ids[start:end]
    b_docs   = docs[start:end]
    b_embtxt = embeds[start:end]
    b_metas  = metas[start:end]

    b_emb = model.encode(b_embtxt, show_progress_bar=False,
                         batch_size=32).tolist()      # cap the model's own batching

    # upsert (not add): re-running an overlapping batch just overwrites the
    # same ids instead of raising a duplicate-id error — makes resume safe.
    collection.upsert(ids=b_ids, documents=b_docs,
                      embeddings=b_emb, metadatas=b_metas)

    # checkpoint: record this batch as done (upsert happened first, so if we
    # reach here the rows are in Chroma)
    with open(PROGRESS_FILE, "a") as pf:
        pf.write(f"{bi}\n")

    if bi % 10 == 0 or end == len(docs):
        print(f"  batch {bi + 1}/{n_batches}  ({end}/{len(docs)})")

    del b_emb, b_docs, b_embtxt, b_metas, b_ids
    gc.collect()

print(f"Done. '{COLLECTION}' now holds {collection.count()} items total.")
os.remove(PROGRESS_FILE)   # clean up on full success