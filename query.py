import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
anthropic_client = anthropic.Anthropic()
client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION = "medical_library"          # MUST match ingest.py
collection = client.get_collection(COLLECTION)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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


def contextualize(query, history):
    if not history:
        return query
    recent = "\n".join(history[-4:])
    rewritten = anthropic_client.messages.create(
        model=HAIKU,
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
    return extract_text(rewritten)


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
    "fabricate citations or page numbers, and say so if you are uncertain."
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
        return ("This question is outside my otorhinolaryngology scope.")

    disclaimer = ("⚠️ Not found in the textbook sources — the following is from "
                  "general ENT knowledge and is not citation-grounded.\n\n")
    return disclaimer + body


def answer(query, user_id="default"):
    history = _histories.setdefault(user_id, [])

    # 1. rewrite follow-ups into standalone questions
    search_query = contextualize(query, history)
    print(f"[searching for: {search_query}]")

    # 2. retrieve across all books and try to answer FROM the sources (Haiku)
    q_emb = embedder.encode([search_query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=8)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[{m.get('source', 'unknown')}, p.{m.get('page', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    )

    prompt = f"""Answer the question using the context below. Answer if the context supports it, even partially.
If the context does not contain enough information, respond with EXACTLY this sentence and nothing else:
"{NO_INFO_MARKER}."
Cite the document name and the page number for each claim you make, like [scott_brown, p.51].
Be concise but do not lose information.

Context:
{context}

Question: {search_query}"""

    message = anthropic_client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = extract_text(message)

    # 3. if the sources were silent, ABANDON rag and escalate to Sonnet
    if NO_INFO_MARKER.lower() in reply.lower():
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
