import os
import chromadb
import anthropic
import json
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
anthropic_client = anthropic.Anthropic()

_HERE = os.path.dirname(os.path.abspath(__file__))
client = chromadb.PersistentClient(path=os.path.join(_HERE, "chroma_db"))

COLLECTION = "medical_library"
collection = client.get_collection(COLLECTION)
print(f"[DB: {os.path.join(_HERE, 'chroma_db')} | count: {collection.count()}]")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
# --- Build BM25 (sparse) index over the whole collection, once at startup ---
def _load_corpus():
    ids, docs, metas = [], [], []
    total = collection.count()
    offset, BATCH = 0, 2000
    while offset < total:
        data = collection.get(include=["documents", "metadatas"],
                              limit=BATCH, offset=offset)
        ids.extend(data["ids"])
        docs.extend(data["documents"])
        metas.extend(data["metadatas"])
        offset += BATCH
    return ids, docs, metas

def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())

print("[building BM25 index...]")
_ALL_IDS, _ALL_DOCS, _ALL_METAS = _load_corpus()
_ID_TO_IDX = {cid: i for i, cid in enumerate(_ALL_IDS)}
_bm25 = BM25Okapi([_tok(d) for d in _ALL_DOCS])
print(f"[BM25 ready: {len(_ALL_IDS)} chunks indexed]")
_histories = {}

HAIKU  = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"              # fallback model for general-knowledge answers
PLANNER = SONNET                       # Sonnet plans retrieval (decompose + reformulate)
# =============================================================================
# TOKEN ACCOUNTING
# Per-request accumulator. Every messages.create call appends via _track();
# app.py calls reset_usage() before a request and pop_usage() after. The whole
# pipeline is serialized under app.py's _lock, so this module-global only ever
# holds one request's calls at a time. Captures ALL calls a request makes —
# planner + generator, plus a fallback (and a wasted Haiku call on escalation).
# =============================================================================
PRICING = {   # USD per 1,000,000 tokens: {input, output}. Verified 2026-08-20.
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "claude-sonnet-5":   {"in": 2.00, "out": 10.00},   # may rise to 3/15 on Sep 1
}
_USAGE = []


def _rate_for(model):
    if model in PRICING:
        return PRICING[model]
    for key, rate in PRICING.items():          # tolerate dated/suffixed strings
        if model.startswith(key) or key in model:
            return rate
    print(f"[usage: no price for '{model}' -> counted at $0]")
    return None


def _track(resp, model):
    """Record token usage + cost for one messages.create call."""
    u = getattr(resp, "usage", None)
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cr  = getattr(u, "cache_read_input_tokens", 0) or 0
    cw  = getattr(u, "cache_creation_input_tokens", 0) or 0
    rate = _rate_for(model)
    if rate:
        per_in, per_out = rate["in"] / 1_000_000, rate["out"] / 1_000_000
        cost = inp * per_in + out * per_out + cr * per_in * 0.10 + cw * per_in * 1.25
    else:
        cost = 0.0
    _USAGE.append({"model": model, "input_tokens": inp, "output_tokens": out,
                   "cost_usd": round(cost, 6)})


def reset_usage():
    _USAGE.clear()


def pop_usage():
    """Sum every call tracked since the last reset, clear, and return a dict
    ready to write to query_logs (scalars + a per-model breakdown)."""
    calls = list(_USAGE)
    _USAGE.clear()
    inp  = sum(c["input_tokens"]  for c in calls)
    out  = sum(c["output_tokens"] for c in calls)
    cost = round(sum(c["cost_usd"] for c in calls), 6)
    by_model = {}
    for c in calls:
        m = by_model.setdefault(c["model"],
                                {"calls": 0, "input_tokens": 0,
                                 "output_tokens": 0, "cost_usd": 0.0})
        m["calls"]         += 1
        m["input_tokens"]  += c["input_tokens"]
        m["output_tokens"] += c["output_tokens"]
        m["cost_usd"]       = round(m["cost_usd"] + c["cost_usd"], 6)
    return {"input_tokens": inp, "output_tokens": out,
            "total_tokens": inp + out, "cost_usd": cost,
            "cost_breakdown": by_model}

# =============================================================================
# ACRONYM EXPANSION (query-time, model-agnostic)
# MiniLM does not resolve medical acronyms: "complications of FESS" lands in a
# weak vector region and retrieves FESS-revision junk, while the spelled-out
# form retrieves correctly. We expand known acronyms before dense encode AND
# BM25 so both retrievers see the meaning. The acronym is KEPT (BM25 still
# matches the literal token in chunks) and the expansion is appended for dense.
# =============================================================================
with open(os.path.join(_HERE, "ent_acronyms.json"), "r", encoding="utf-8") as _f:
    ACRONYMS = json.load(_f)

# Lowercase forms we will NOT expand: real English words + ambiguous short
# strings. Uppercase always expands regardless of this set.
LOWERCASE_NO_EXPAND = {"an", "ar", "pet", "spa", "us"}

_ACR_KEYS = sorted(ACRONYMS.keys(), key=len, reverse=True)  # longest-match first
_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ACR_KEYS) + r")\b", re.IGNORECASE
)
_ACR_UPPER = {k.upper(): v for k, v in ACRONYMS.items()}


def _should_expand(matched):
    if matched.isupper():
        return True                                    # uppercase always expands
    return matched.lower() not in LOWERCASE_NO_EXPAND   # lowercase: unless blocked


def expand_acronyms(query):
    """'complications of FESS' -> 'complications of FESS (functional endoscopic sinus surgery)'"""
    seen = set()

    def _sub(m):
        matched = m.group(1)
        canon = matched.upper()
        if not _should_expand(matched) or canon in seen:
            return matched
        seen.add(canon)
        return f"{matched} ({_ACR_UPPER[canon]})"

    return _ACRONYM_RE.sub(_sub, query)


# =============================================================================
# DETERMINISTIC BOOK ROUTING
# Only restrict the search when the user EXPLICITLY names a book. No LLM, no
# variance — a wrong restriction silently drops the right chunk, so this stays
# deterministic. Maps user-facing spellings -> the actual metadata `source`
# value. NOTE: the corpus stores Shambaugh's source as the misspelling
# "shaumbaugh", so the user's correct "shambaugh" must map to that.
# =============================================================================
BOOK_ALIASES = {
    "cummings":         "cummings",
    "scott-brown":      "scott",
    "scott brown":      "scott",
    "scott":            "scott",
    "stell and maran":  "stell",
    "stell & maran":    "stell",
    "stell":            "stell",
    "shambaugh":        "shaumbaugh",   # user spelling -> metadata spelling
    "shaumbaugh":       "shaumbaugh",
}
_ALIASES_SORTED = sorted(BOOK_ALIASES.keys(), key=len, reverse=True)


def detect_books(question):
    """Return a list of source strings if the user explicitly named book(s),
    else None (search all books)."""
    q = question.lower()
    found = []
    for alias in _ALIASES_SORTED:
        if alias in q:
            src = BOOK_ALIASES[alias]
            if src not in found:
                found.append(src)
            q = q.replace(alias, " ")   # consume so shorter aliases can't re-match
    return found or None


# =============================================================================
# SONNET QUERY PLANNER  (Sonnet plans retrieval, Haiku writes the answer)
# Two kinds of decomposition, each gated:
#   - TOPIC   (multi-entity "X vs Y")            -> split by entity
#   - FACET   (single broad UNSCOPED entity "X") -> optionally split by clinical
#             facet (diagnosis/management/complications) ONLY if the user did
#             not already pick a facet
#   - NEITHER (already focused: "treatment of X", "complications of FESS") -> one search
# Sonnet reformulates into textbook terminology but must NOT inject specific
# drugs/facts/sub-topics the user didn't ask about — it rephrases the question,
# it does not pre-answer it. That preserves the "books decide, not the model"
# guarantee the citations rest on. Book selection is NOT Sonnet's job — that is
# handled deterministically by detect_books().
# =============================================================================
PLAN_TOOL = {
    "name": "search_plan",
    "strict": True,
    "description": (
        "Plan how to answer the user's ENT question. Return (1) the standalone "
        "question the assistant should ANSWER, with any follow-up references "
        "resolved from the conversation, and (2) the search sub-queries used to "
        "RETRIEVE textbook chunks."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "standalone_question": {
                "type": "string",
                "description": (
                    "The complete, self-contained question the assistant should "
                    "answer, phrased as a real question a person would ask. "
                    "Resolve follow-up references from the conversation: "
                    "'give more complications' after a FESS question becomes "
                    "'What are the complications of FESS not already covered "
                    "above?'. This is what the ANSWERING model reads — it must "
                    "be a full question, NOT a search fragment or keyword list."
                ),
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-4 self-contained search strings in formal textbook "
                    "terminology (these can be terse). Prefer FEWER, higher-"
                    "quality sub-queries — do not manufacture a marginal extra "
                    "facet just to add one. Return ONE for a focused "
                    "question. For a BROAD question (complications / types / "
                    "causes / features / management of something, or a bare "
                    "entity name), break it into the natural sub-topics a "
                    "textbook covers as separate sections — one query per "
                    "sub-topic — so each lands in its own part of the corpus. "
                    "For a MULTI-TOPIC question (comparison / two asks), one "
                    "query per topic. Expand acronyms; use textbook vocabulary. "
                    "For a 'give more' follow-up, target the sub-topics NOT "
                    "already covered in the previous answer."
                ),
            },
            "in_scope": {
                "type": "boolean",
                "description": (
                    "Whether the question is answerable within ENT "
                    "(otorhinolaryngology / head & neck, audiology and directly "
                    "related topics). Interpret ambiguous questions in the ENT "
                    "frame and set true — 'what antibiotics are used' means in "
                    "ENT, 'NICE guidelines' means the ENT-relevant ones. Set "
                    "FALSE only when the question is clearly a non-ENT medical "
                    "field with no reasonable ENT reading (orthopedic fractures, "
                    "cardiac conditions, obstetrics). When in doubt, true."
                ),
            },
        },
        "required": ["standalone_question", "sub_queries", "in_scope"],
    },
}

PLANNER_SYSTEM = (
    "You are the retrieval planner for a citation-grounded ENT (otolaryngology) "
    "exam-prep assistant. The corpus is four standard ENT textbooks, and every "
    "sub-query is searched across ALL of them together — you do NOT choose which "
    "book to search and must NOT tailor wording toward any single one. You have "
    "strong ENT knowledge; use it to plan retrieval that yields the most "
    "COMPLETE, well-grounded set of chunks. Each answer cites specific textbook "
    "pages, so the chunks you cause to be retrieved determine answer quality.\n\n"
    "Return three things via search_plan:\n\n"
    "A) standalone_question — the full question the assistant should ANSWER, "
    "with follow-up references resolved from the conversation. If the user says "
    "'give more complications' after a FESS answer, resolve it to 'What are the "
    "complications of FESS not already covered above?'. It MUST read as a "
    "complete question, never a keyword fragment.\n\n"
    "B) sub_queries — the searches that retrieve chunks:\n"
    "- FOCUSED question (one topic + one facet) -> ONE query. "
    "e.g. 'otoscopic appearance of cholesteatoma'.\n"
    "- MULTI-TOPIC (comparison / two asks) -> one query PER topic. "
    "e.g. 'CSOM vs cholesteatoma' -> a query for each.\n"
    "- BROAD (complications / types / causes / features / management of X, or a "
    "bare entity) -> break into the natural sub-topics a textbook covers as "
    "SEPARATE SECTIONS, one query each, using your ENT knowledge to choose them. "
    "e.g. 'complications of FESS' -> hemorrhagic complications of endoscopic "
    "sinus surgery; orbital complications of endoscopic sinus surgery; "
    "intracranial and CSF leak complications of endoscopic sinus surgery; local "
    "complications of endoscopic sinus surgery (synechiae, nasolacrimal injury, "
    "anosmia).\n"
    "Prefer FEWER, well-chosen sub-topics (typically 3-4 max). Do not invent a "
    "thin extra facet just to pad the list — a marginal sub-query pulls weak "
    "chunks that can crowd out strong ones.\n"
    "- 'GIVE MORE' follow-up -> target the sub-topics NOT already covered in the "
    "previous answer shown in the conversation.\n\n"
    "Reformulate each sub-query for RETRIEVAL, which matches literal words:\n"
    "- Expand acronyms (FESS -> functional endoscopic sinus surgery).\n"
    "- Convert lay terms to medical terms (ringing in ears -> tinnitus).\n"
    "- Prefer the COMMON clinical term the textbook actually prints, and include "
    "key synonyms, because a formal term alone can miss the passage. Use "
    "'bleeding epistaxis' not just 'hemorrhagic'; 'CSF leak cerebrospinal fluid' "
    "not just one; include both a term and its close synonyms in the same "
    "sub-query when they differ.\n\n"
    "Two rules to hold:\n"
    "1. PRESERVE SCOPE. If the user narrowed to one facet ('treatment of X'), "
    "decompose WITHIN it (medical vs surgical management) but do not wander into "
    "other facets.\n"
    "2. Search by CATEGORY, not by specific answer. Name the sub-topic to look "
    "in ('medical management of otosclerosis'), not the specific facts you "
    "expect to find. You decide WHERE to look; the textbook decides WHAT is there."
    "\n\nC) in_scope — whether this question belongs to ENT (ear, nose, throat, "
    "head & neck, audiology and directly related topics). Interpret ambiguous "
    "questions in the ENT frame and set it true; set it FALSE only when the "
    "question is clearly a non-ENT medical field with no reasonable ENT reading "
    "(orthopedic fractures, cardiac conditions, obstetrics). When in doubt, true. "
    "Always still return a standalone_question and sub_queries even when it is false."
)

def plan_query(question, history=None):
    """Ask Sonnet for a search plan. Returns (standalone_question, sub_queries).
    standalone_question is what generation answers; sub_queries drive retrieval.
    Falls back to (question, [question]) if anything goes wrong."""
    if history:
        recent = "\n".join(history[-6:])   # include prior answer so 'more' works
        content = (f"Conversation so far:\n{recent}\n\n"
                   f"Current question: {question}")
    else:
        content = question
    try:
        msg = anthropic_client.messages.create(
            model=PLANNER,
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=PLANNER_SYSTEM,
            tools=[PLAN_TOOL],
            tool_choice={"type": "tool", "name": "search_plan"},
            messages=[{"role": "user", "content": content}],
        )
        _track(msg, PLANNER)
        plan = next((b.input for b in msg.content
                     if getattr(b, "type", None) == "tool_use"), None)
        if not plan:
            return question, [question], True
        standalone = (plan.get("standalone_question") or question).strip()

        # scope flag: default True when absent/None so a missing field never refuses
        raw_scope = plan.get("in_scope")
        in_scope = True if raw_scope is None else bool(raw_scope)

        raw_subs = plan.get("sub_queries")
        # schema is NOT strictly enforced: a malformed/truncated tool call can
        # return sub_queries as a STRING, which iterates into characters and
        # explodes retrieval. Wrap a string into a single-item list.
        if isinstance(raw_subs, str):
            raw_subs = [raw_subs]
        if not isinstance(raw_subs, list):
            raw_subs = [standalone]
        subs = [s.strip() for s in raw_subs if isinstance(s, str) and s.strip()]

        # if we still got a pile of 1-char fragments (explosion signature),
        # discard and fall back to the standalone question — stays GROUNDED.
        if subs and sum(len(s) for s in subs) / len(subs) < 3:
            print("[planner sub_queries fragmented -> using standalone]")
            subs = [standalone]

        return standalone, (subs if subs else [standalone]), in_scope
    except Exception as e:
        print(f"[planner error -> single search: {e}]")
        return question, [question], True


def extract_text(msg):
    """
    Return the assistant's text from a response. Some models (e.g. Sonnet with
    extended thinking) put a ThinkingBlock first, so content[0] may NOT be text.
    Join all blocks that actually have a .text attribute.
    """
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()

# Exact phrase we instruct Haiku to emit when the corpus can't answer.
# Detecting it is how we decide to escalate to Sonnet.
NO_INFO_MARKER = "The provided sources do not address this"


# --- citation + table-cleanup helpers (added for the web frontend) ---------
# Matches the frontend's own regex so markers and citation objects line up.
CITE_RE = re.compile(r"\[([a-z]+),?\s*p\.?\s*(\d+)\]", re.I)


def _is_ruler_line(line):
    """A markdown separator line like |---|---| — noise in the source drawer."""
    s = line.strip()
    return bool(s) and set(s) <= set("|-: ")


def collapse_dup_cells(text):
    """
    Cosmetic repair for Docling's column-duplication artifact: a merged or
    single-column cell splayed across N columns as identical copies, e.g.
        | TX | Unable to assess | Unable to assess | Unable to assess |
    collapses to
        TX | Unable to assess
    Facts are never dropped — only exact repeats of the SAME value in a row.
    Trivial short cells (M0, N1, 5.0) are left alone so real staging rows and
    numeric tables are untouched. Used only when we hand a chunk to the drawer.
    """
    if "|" not in text:
        return text.strip()
    out = []
    for line in text.split("\n"):
        if _is_ruler_line(line):
            continue
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            non_empty = [c for c in cells if c]
            if len(non_empty) >= 2 and len(set(non_empty)) == 1:
                # whole row was one value splayed across every column
                line = non_empty[0]
            else:
                collapsed = []
                for c in cells:
                    # drop a non-trivial cell that just repeats the previous one
                    if collapsed and c == collapsed[-1] and len(c) >= 20:
                        continue
                    collapsed.append(c)
                line = " | ".join(c for c in collapsed if c)
        out.append(line)
    return "\n".join(l for l in out if l.strip()).strip()


def build_citations(answer_text, picked):
    """
    Turn the [book, p.N] markers Haiku wrote into the citation objects the
    frontend needs: {id, book, page, chunk_text}. One entry per distinct
    (book, page) actually cited, in order of first appearance, backed by the
    highest-ranked retrieved chunk for that page. Markers with no matching
    retrieved chunk are skipped (the chip still renders via the frontend's
    {book,page} fallback, just without a source passage).
    """
    by_key = {}
    for rec in picked:  # picked is already rrf-sorted, so first = best
        m = rec["meta"]
        key = (str(m.get("source", "")).lower(), str(m.get("page", "")))
        by_key.setdefault(key, rec)

    seen, cites = set(), []
    for mo in CITE_RE.finditer(answer_text):
        book, page = mo.group(1).lower(), mo.group(2)
        key = (book, page)
        if key in seen:
            continue
        seen.add(key)
        rec = by_key.get(key)
        if not rec:
            continue
        cites.append({
            "id": rec["id"],
            "book": book,
            "page": int(page),
            "chunk_text": collapse_dup_cells(rec["doc"]),
        })
    return cites


OUT_OF_SCOPE_MARKER = "OUT_OF_SCOPE"
OUT_OF_SCOPE_REPLY = (
    "This question is outside my scope. I'm an ENT (ear, nose, throat, "
    "head & neck) reference assistant and can't help with topics outside "
    "otorhinolaryngology."
)
FALLBACK_SYSTEM = (
    "You are an ENT (otorhinolaryngology / head and neck surgery) reference "
    "assistant. Your scope is ENT: the ear, nose, throat, head and neck, "
    "audiology, and directly related topics.\n"
    "The user's question could not be answered from the indexed ENT textbook, so "
    "you may answer from your own general medical knowledge.\n"
    "IMPORTANT: the user is talking to an ENT assistant, so interpret ambiguous "
    "questions in the ENT context. For example, 'what are the NICE guidelines' means "
    "'what are the NICE guidelines relevant to ENT', and 'what antibiotics are used' "
    "means 'in ENT'. Answer such questions within the ENT frame.\n"
    f"ONLY refuse if the question is CLEARLY about a non-ENT medical field with no "
    f"reasonable ENT interpretation (e.g. orthopedic fractures, cardiac conditions, "
    f"obstetrics). In that case respond with EXACTLY this token and nothing else: "
    f"{OUT_OF_SCOPE_MARKER}\n"
    "When you answer, be accurate and concise, focus on the ENT angle, do NOT "
    "fabricate citations or page numbers, and say so if you are uncertain.\n"
)


def sonnet_fallback(question):
    """
    Corpus was silent -> answer from Sonnet's own knowledge, but ONLY within ENT
    scope, and clearly flagged as NOT from the textbook sources. Out-of-ENT
    questions are refused rather than answered.
    """
    msg = anthropic_client.messages.create(
        model=SONNET,
        max_tokens=1500,
        system=FALLBACK_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    _track(msg, SONNET)
    body = extract_text(msg)

    if OUT_OF_SCOPE_MARKER in body:
        return OUT_OF_SCOPE_REPLY

    disclaimer = ("⚠️ Not found in the textbook sources — the following is from "
                  "general ENT knowledge, is not citation-grounded, and should be "
                  "verified against a primary source.\n\n")
    return disclaimer + body

def retrieve_filtered(query_text, raw_k=60, keep=9, rrf_k=60, sources=None):
    query_text = expand_acronyms(query_text)   # acronym-blindness fix (both retrievers)

    # optional book filter: only restrict when detect_books() found an explicit mention
    where = {"source": {"$in": sources}} if sources else None

    # --- dense (MiniLM / vector) ---
    q_emb = embedder.encode([query_text]).tolist()
    dres = collection.query(query_embeddings=q_emb, n_results=raw_k,
                            where=where,
                            include=["documents", "metadatas", "distances"])
    dense_ids = dres["ids"][0]
    dense_lookup = {cid: {"doc": dres["documents"][0][i],
                          "meta": dres["metadatas"][0][i],
                          "dist": dres["distances"][0][i]}
                    for i, cid in enumerate(dense_ids)}

    # --- sparse (BM25 / keyword) ---
    scores = _bm25.get_scores(_tok(query_text))
    # widen the BM25 pull when filtering, so post-filtering to chosen books
    # doesn't starve the fused candidate set.
    bm_k = raw_k * (3 if sources else 1)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i],
                     reverse=True)[:bm_k]
    sparse_ids = [_ALL_IDS[i] for i in top_idx]
    if sources:
        # BM25 has no `where`; post-filter its hits to the chosen books so the
        # fusion stays consistent with the (already-filtered) dense side.
        allowed = set(sources)
        sparse_ids = [cid for cid in sparse_ids
                      if _ALL_METAS[_ID_TO_IDX[cid]].get("source") in allowed][:raw_k]

    # --- Reciprocal Rank Fusion: combine the two rankings by position ---
    rrf = {}
    DENSE_W, SPARSE_W = 1.0, 0.5  # give BM25 half the vote
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + DENSE_W / (rrf_k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + SPARSE_W / (rrf_k + rank + 1)

    # Table boost: tables embed well (caption-weighted) and score high on dense,
    # but get out-competed by prose in the fused+merged+capped pipeline. Give any
    # table that ALREADY ranked decently on dense a small additive nudge so a
    # genuinely-relevant table clears FINAL_CAP. Irrelevant tables score low on
    # dense and never enter dense_lookup, so they get no boost.
    TABLE_BOOST = 0.010
    for cid in list(rrf.keys()):
        rec = dense_lookup.get(cid)
        if rec and rec["meta"].get("type") == "table":
            rrf[cid] += TABLE_BOOST

    fused_ids = sorted(rrf, key=lambda c: rrf[c], reverse=True)

    out = []
    for cid in fused_ids:
        if cid in dense_lookup:
            doc, meta = dense_lookup[cid]["doc"], dense_lookup[cid]["meta"]
        else:                                   # BM25-only hit
            idx = _ID_TO_IDX.get(cid)
            if idx is None:
                continue
            doc, meta = _ALL_DOCS[idx], _ALL_METAS[idx]
        # Known-corrupt Weber/Rinne table (semantic cross-wiring, clinically
        # unsafe): drop the whole p83 table family until re-extraction.
        if cid.startswith("cummings_p83_t0_28"):
            continue
        # same stub filter as before
        is_prose_stub = (meta.get("type") == "prose"
                         and not meta.get("is_figure")
                         and len(doc.strip()) < 80)
        if is_prose_stub:
            continue
        out.append({"id": cid, "doc": doc, "meta": meta, "rrf": rrf[cid]})
        if len(out) >= keep:
            break
    return out
def answer(query, user_id="default"):
    history = _histories.setdefault(user_id, [])

    # 1. deterministic book routing: restrict ONLY if the user named a book
    books = detect_books(query)
    if books:
        print(f"[book filter: user named {books}]")

    # 2. Sonnet plans in ONE call: resolves the follow-up into a standalone
    #    question (what Haiku answers) AND decomposes into retrieval sub-queries.
    #    This replaces both contextualize() and decompose().
    search_query, subqueries, in_scope = plan_query(query, history)
    if not in_scope:
        reply = OUT_OF_SCOPE_REPLY
        history.append(f"User: {query}")
        history.append(f"Assistant: {reply}")
        if len(history) > 20:
            del history[:-20]
        return reply, [], False, search_query
    print(f"[answer question: {search_query}]")
    print(f"[plan: {subqueries}]")

    # 3. retrieve per sub-query, then merge with PER-SUB-QUERY QUOTAS.
    #    Problem this fixes: a global RRF sort lets one "loud" sub-query (high
    #    RRF chunks) fill the cap and drown other facets (e.g. Q4 cholesteatoma:
    #    the definition facet crowded out otoscopy + management). Fix: guarantee
    #    each sub-query a reserved share of the final slots, so every facet the
    #    planner identified reaches the answer. A QUALITY FLOOR means a sub-query
    #    only claims reserved slots if its chunks are actually good — so the
    #    planner gains nothing from padding the plan with weak extra sub-queries.
    SUBQUERY_QUOTA = 2 if len(subqueries) <= 4 else 1  # fewer reserved when many subs
    QUOTA_FLOOR = 0.015      # a chunk must clear this RRF to claim a reserved slot
    FINAL_CAP = 16
    per_keep = 16 if len(subqueries) == 1 else 9

    # retrieve per sub-query, keep grouped (each in its own RRF-sorted list)
    groups = []
    for sq in subqueries:
        hits = retrieve_filtered(sq, raw_k=60, keep=per_keep, sources=books)
        groups.append(hits)   # already rrf-sorted by retrieve_filtered

    # PASS 1 - reservation: each sub-query reserves up to SUBQUERY_QUOTA of its
    # best chunks that clear the floor and aren't already taken. A sub-query with
    # nothing above floor reserves nothing (anti-padding).
    picked_by_id = {}
    for hits in groups:
        taken = 0
        for rec in hits:
            if taken >= SUBQUERY_QUOTA:
                break
            cid = rec["id"]
            if cid in picked_by_id:          # already reserved by an earlier sub-query
                continue
            if rec["rrf"] < QUOTA_FLOOR:      # below quality floor -> don't reserve
                continue
            picked_by_id[cid] = rec
            taken += 1

    # PASS 2 - global fill: pool everything not yet picked, sort by RRF, fill the
    # remaining slots up to FINAL_CAP. Keeps best global chunks AND lets a rich
    # facet earn extra representation beyond its reserved quota.
    pool = {}
    for hits in groups:
        for rec in hits:
            cid = rec["id"]
            if cid in picked_by_id:
                continue
            if cid not in pool or rec["rrf"] > pool[cid]["rrf"]:
                pool[cid] = rec
    fill = sorted(pool.values(), key=lambda r: r["rrf"], reverse=True)
    for rec in fill:
        if len(picked_by_id) >= FINAL_CAP:
            break
        picked_by_id[rec["id"]] = rec

    # final ordering by RRF for the context (reserved + filled together)
    picked = sorted(picked_by_id.values(), key=lambda r: r["rrf"],
                    reverse=True)[:FINAL_CAP]
    docs = [r["doc"] for r in picked]
    metas = [r["meta"] for r in picked]

    context = "\n\n".join(
        f"[{m.get('source', 'unknown')}, p.{m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    print(f"[context: {len(docs)} chunks, {len(context)} chars]")
    print(f"[sources: {[(m.get('source'), m.get('page')) for m in metas]}]")
    if not docs:
        print("[empty context -> Sonnet fallback]")
        reply = sonnet_fallback(search_query)
        history.append(f"User: {query}")
        history.append(f"Assistant: {reply}")
        if len(history) > 20:
            del history[:-20]
        return reply, [], False, search_query
    prompt= f"""You are answering a medical question using retrieved textbook excerpts.

The context below contains excerpts from one or more ENT textbooks. Your job is to
answer the question by synthesizing across these excerpts.

IMPORTANT:
- The answer may be SPREAD ACROSS several excerpts rather than stated in one place.
  Gather relevant points from all of them and combine into a complete answer.
- Partial or indirect relevance still counts. If excerpts discuss the topic, USE them.
- Tables/lists formatted with pipes still count as content — read and present them.
- Do NOT refuse just because there is no single excerpt that states the whole answer,
  or because the phrasing differs from the question.

Refuse ONLY if the context is genuinely, entirely unrelated to the question — i.e.
NONE of the excerpts touch the topic at all. In that case, respond with EXACTLY this
sentence and nothing else:
"{NO_INFO_MARKER}."

When you answer:
- Cite the document name and page for each claim. Cite each source in its own
  bracket, with one book and one page per bracket:
    [cummings, p.3431] [shaumbaugh, p.1042]   correct
  Never combine two sources in one bracket, and never use a page range. Write
  separate brackets instead:
    [cummings, p.3431; shaumbaugh, p.1042]    wrong  (two sources grouped)
    [shaumbaugh, p.742-743]                   wrong  (page range)
- Be concise but complete; do not drop information.
- If the question asks how two conditions DIFFER (X vs Y), lead with the
  concrete point-by-point contrast — the specific distinguishing features a
  clinician uses at the bedside (e.g. discharge character, perforation site,
  examination findings, complication risk). Put any "these overlap / the
  classification is nuanced" caveat LAST, not first. Do not answer a contrast
  question with a general classification lecture.
- If an excerpt contains a pipe-formatted table, REPRODUCE it as a markdown
  table: keep the pipes, and include the |---| header-separator row so it
  renders as a grid. Preserve the column meaning; do not invent columns.
  Use an indented list only for non-tabular grouped data, not for real tables.

Context:
{context}

Question: {search_query}"""

    message = anthropic_client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    _track(message, HAIKU)
    reply = extract_text(message)

    # 3. escalate to Sonnet ONLY when Haiku actually refused. The reliable signal
    #    is the marker appearing at the START of the reply (Haiku leads with it when
    #    it has nothing). A marker buried mid-answer means it gave real content, so
    #    we keep the RAG answer.
    stripped = reply.strip()
    core = stripped.lstrip('"# ').strip()
    is_refusal = (core.lower().startswith(NO_INFO_MARKER.lower())
                  or (NO_INFO_MARKER.lower() in stripped.lower() and len(stripped) < 350))
    if is_refusal:
        print("[no info in sources -> escalating to Sonnet]")
        reply = sonnet_fallback(search_query)
        citations = []                    # fallback has no grounded citations
        grounded = False
    else:
        citations = build_citations(reply, picked)
        grounded = True

    # 4. record the turn for this user's follow-up context
    history.append(f"User: {query}")
    history.append(f"Assistant: {reply}")
    if len(history) > 20:
        del history[:-20]

    return reply, citations, grounded, search_query


if __name__ == "__main__":
    while True:
        que = input("\nAsk (or 'quit'): ")
        if que.lower() == "quit":
            break
        text, cites, grounded, resolved = answer(que)
        print("\n" + text)
        print("resolved:", resolved)
        print("grounded:", grounded)
        print("citations:", [(c["book"], c["page"]) for c in cites])
