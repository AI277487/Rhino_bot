import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
anthropic_client = anthropic.Anthropic()
client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION = "medical_library"          # MUST match ingest.py
collection = client.get_collection(COLLECTION)

# SAME model as ingest.py, so query vectors and stored vectors are directly
# comparable. We embed the query ourselves and pass query_embeddings=, rather
# than query_texts= (which would make Chroma embed with its own default model).
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Per-user conversation history, keyed by user id, so users NEVER share context.
_histories = {}


def contextualize(query, history):
    if not history:
        return query                      # first question — nothing to resolve
    recent = "\n".join(history[-4:])      # last couple of exchanges
    rewritten = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content":
            f"""Given the conversation below, rewrite the follow-up question into a
standalone question that names the specific topic (resolve words like "it", "this",
"they"). If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation:
{recent}

Follow-up question: {query}"""}],
    )
    return rewritten.content[0].text.strip()


def answer(query, user_id="default"):
    history = _histories.setdefault(user_id, [])

    # 1. rewrite follow-ups into standalone questions BEFORE retrieving
    search_query = contextualize(query, history)
    print(f"[searching for: {search_query}]")

    # 2. retrieve across ALL books (no source filter — search-all by design)
    q_emb = embedder.encode([search_query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=8)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    # build context, tagging each chunk with its source + page
    context = "\n\n".join(
        f"[{m.get('source', 'unknown')}, p.{m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )

    prompt = f"""Answer the question using the context below. Answer if the context supports it, even partially.
If the context does not contain enough information, say
"The provided sources do not address this." Cite the document name and the page number
for each claim you make, like [scott_brown, p.51].
Be concise but do not lose information.

Context:
{context}

Question: {search_query}"""

    message = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = message.content[0].text

    # 3. collect sources (book + page) for the caller to render as citations
    sources = [
        {"source": m.get("source"), "page": m.get("page")}
        for m in metas
    ]

    # 4. record this turn so the NEXT follow-up (from THIS user) has context
    history.append(f"User: {query}")
    history.append(f"Assistant: {reply}")
    if len(history) > 20:                 # cap memory so it can't grow forever
        del history[:-20]

    return reply, sources


# Only runs on `python query.py` directly (terminal testing). The bot imports
# answer() and never touches this block.
if __name__ == "__main__":
    while True:
        que = input("\nAsk (or 'quit'): ")
        if que.lower() == "quit":
            break
        text, srcs = answer(que)
        print("\n" + text)
        print("sources:", srcs)
