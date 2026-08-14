import os
import chromadb
import anthropic
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


CONTEXTUALIZE_TOOL = {
    "name": "emit_search_query",
    "description": "Return the standalone search query for the follow-up question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": ("The follow-up rewritten as a standalone question. "
                                "If it introduces a new unrelated topic, return it "
                                "verbatim. No commentary, just the question."),
            }
        },
        "required": ["query"],
    },
}


def contextualize(query, history):
    if not history:
        return query

    # skip the LLM entirely for clearly-standalone questions (no back-reference)
    words = query.lower().split()
    has_backref = any(w in words for w in
                      ("it", "its", "this", "that", "they", "them", "these",
                       "those", "he", "she", "his", "her", "their", "one"))
    if not has_backref and len(words) >= 3:
        return query

    recent = "\n".join(history[-4:])
    try:
        msg = anthropic_client.messages.create(
            model=HAIKU,
            max_tokens=150,
            tools=[CONTEXTUALIZE_TOOL],
            tool_choice={"type": "tool", "name": "emit_search_query"},
            messages=[{"role": "user", "content":
                f"""Rewrite the follow-up into a standalone search query using the conversation for context.

Rules:
- If it refers back to the prior topic (it/this/they/that, or clearly continuing), rewrite it to name that specific topic.
- If it introduces a NEW, unrelated topic, return it UNCHANGED — do NOT attach the prior topic.

Conversation:
{recent}

Follow-up: {query}"""}],
        )
        out = next((b.input.get("query") for b in msg.content
                    if getattr(b, "type", None) == "tool_use"), None)
        return out.strip() if out else query
    except Exception:
        return query          # any API hiccup -> just search the raw query
DECOMPOSE_TOOL = {
    "name": "decompose_query",
    "description": (
        "Break an ENT question into focused retrieval sub-queries. For comparative "
        "or multi-topic questions (differentiate X from Y, X vs Y, difference "
        "between X and Y), produce ONE self-contained sub-query per topic, each "
        "naming that topic explicitly and stating what is asked (its management, "
        "its otoscopic appearance, etc.). For a simple single-topic question, "
        "return the question unchanged as the only item."
        "For MANAGEMENT/TREATMENT questions, keep the condition name as the primary term "
        "and append a few general treatment-category words (e.g. 'treatment, medical "
        "management, drug therapy, prophylaxis, surgical options') to steer retrieval "
        "toward therapy sections. Do NOT list specific named drugs unless they appear in "
        "the user's question — use only generic category terms."

    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subqueries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-4 focused, self-contained search queries.",
            }
        },
        "required": ["subqueries"],
    },
}


def decompose(query):
    try:
        msg = anthropic_client.messages.create(
            model=HAIKU,
            max_tokens=250,
            tools=[DECOMPOSE_TOOL],
            tool_choice={"type": "tool", "name": "decompose_query"},
            messages=[{"role": "user", "content":
                f"Split this ENT question into focused retrieval sub-queries.\n\n"
                f"Question: {query}"}],
        )
        subs = next((b.input.get("subqueries") for b in msg.content
                     if getattr(b, "type", None) == "tool_use"), None)
        subs = [s.strip() for s in (subs or []) if s and s.strip()]
        return subs if subs else [query]
    except Exception:
        return [query]        # any hiccup -> single-query behaviour, unchanged

OUT_OF_SCOPE_MARKER = "OUT_OF_SCOPE"

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
    "Do NOT use markdown tables or pipe characters (they don't render in Telegram); "
    "present any tabular data as a readable indented list grouped by category."
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
    body = extract_text(msg)

    if OUT_OF_SCOPE_MARKER in body:
        return ("This question is outside my scope. I'm an ENT (ear, nose, throat, "
                "head & neck) reference assistant and can't help with topics outside "
                "otorhinolaryngology.")

    disclaimer = ("⚠️ Not found in the textbook sources — the following is from "
                  "general ENT knowledge, is not citation-grounded, and should be "
                  "verified against a primary source.\n\n")
    return disclaimer + body

def retrieve_filtered(query_text, raw_k=60, keep=8, rrf_k=60):
    # --- dense (MiniLM / vector) ---
    q_emb = embedder.encode([query_text]).tolist()
    dres = collection.query(query_embeddings=q_emb, n_results=raw_k,
                            include=["documents", "metadatas", "distances"])
    dense_ids = dres["ids"][0]
    dense_lookup = {cid: {"doc": dres["documents"][0][i],
                          "meta": dres["metadatas"][0][i],
                          "dist": dres["distances"][0][i]}
                    for i, cid in enumerate(dense_ids)}

    # --- sparse (BM25 / keyword) ---
    scores = _bm25.get_scores(_tok(query_text))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i],
                     reverse=True)[:raw_k]
    sparse_ids = [_ALL_IDS[i] for i in top_idx]

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

    # 1. rewrite follow-ups into standalone questions
    search_query = contextualize(query, history)
    print(f"[searching for: {search_query}]")

    # 2. retrieve across all books and try to answer FROM the sources (Haiku)
    # 2. decompose, retrieve per sub-query, merge (dedupe by chunk id, keep best dist)
    subqueries = decompose(search_query)
    print(f"[subqueries: {subqueries}]")

    per_keep = 15 if len(subqueries) == 1 else 8
    merged = {}
    for sq in subqueries:
        for rec in retrieve_filtered(sq, raw_k=60, keep=per_keep):
            cid = rec["id"]
            if cid not in merged or rec["rrf"] > merged[cid]["rrf"]:  # <- changed
                merged[cid] = rec

    FINAL_CAP = 15
    picked = sorted(merged.values(), key=lambda r: r["rrf"],  # <- changed
                    reverse=True)[:FINAL_CAP]
    docs = [r["doc"] for r in picked]
    metas = [r["meta"] for r in picked]

    context = "\n\n".join(
        f"[{m.get('source', 'unknown')}, p.{m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    print(f"[context: {len(docs)} chunks, {len(context)} chars]")
    print(f"[sources: {[(m.get('source'), m.get('page')) for m in metas]}]")

    context = "\n\n".join(
        f"[{m.get('source', 'unknown')}, p.{m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    print(f"[context: {len(docs)} chunks, {len(context)} chars]")  # <-- add
    print(f"[sources: {[(m.get('source'), m.get('page')) for m in metas]}]")
    if not docs:
        print("[empty context -> Sonnet fallback]")
        reply = sonnet_fallback(search_query)
        history.append(f"User: {query}")
        history.append(f"Assistant: {reply}")
        if len(history) > 20:
            del history[:-20]
        return reply, []
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
- Cite the document name and page for each claim, like [cummings, p.318].
- Be concise but complete; do not drop information.
- Re-present any pipe-formatted table as an indented list, never with pipes, e.g.:
  Infective:
   - Tuberculosis (Mycobacterium tuberculosis)
  Inflammatory:
   - Sarcoidosis

Context:
{context}

Question: {search_query}"""

    message = anthropic_client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
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
        sources = []                      # fallback has no citations
    else:
        sources = [
            {"source": m.get("source"), "page": m.get("page")}
            for m in metas
        ]

    # 4. record the turn for this user's follow-up context
    history.append(f"User: {query}")
    history.append(f"Assistant: {reply}")
    if len(history) > 20:
        del history[:-20]

    return reply, sources


if __name__ == "__main__":
    while True:
        que = input("\nAsk (or 'quit'): ")
        if que.lower() == "quit":
            break
        text, srcs = answer(que)
        print("\n" + text)
        print("sources:", srcs)
