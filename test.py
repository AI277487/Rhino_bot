"""
table_one.py -- ONE run per question, everything printed in that single run.

Per question (one answer() call, so one paste has the whole diagnosis):
  - the answer, and whether a pipe table survived into it
  - the RAW cited chunks straight from ChromaDB (what Haiku actually saw),
    with type + whether the raw text is a clean pipe grid

No second script, no second paid run. Fetching chunks is local (free); the only
API cost is one answer() per question.

Run:
  python table_one.py > table_run.txt
"""
import query
from query import answer

QUESTIONS = [
    "What is the House-Brackmann grading of facial nerve function?",
    "Outline the T staging of glottic carcinoma.",
]


def pipe_rows(text):
    return [ln for ln in text.split("\n") if ln.count("|") >= 2]


def fetch_raw(cid):
    try:
        g = query.collection.get(ids=[cid], include=["documents", "metadatas"])
        if g and g["ids"]:
            return g["documents"][0], g["metadatas"][0]
    except Exception as e:
        return None, {"_error": str(e)}
    return None, None


for i, q in enumerate(QUESTIONS, 1):
    reply, cites, grounded = answer(q, user_id=f"tbl_{i}")
    rows = pipe_rows(reply)

    print("=" * 80)
    print(f"Q{i}: {q}")
    print(f"grounded={grounded}  answer_pipe_rows={len(rows)}  "
          + ("TABLE SURVIVED" if rows else "STILL FLATTENED (no table)"))
    print(f"cited: {[(c['book'], c['page']) for c in cites]}")
    print("-" * 80)
    print("ANSWER:")
    print(reply)

    print("-" * 80)
    print("RAW CITED CHUNKS (what Haiku saw):")
    for c in cites:
        raw, meta = fetch_raw(c["id"])
        print("  " + "-" * 76)
        print(f"  id={c['id']}  {c['book']} p.{c['page']}")
        if raw is None:
            print(f"    [could not fetch: {meta}]")
            continue
        rr = pipe_rows(raw)
        print(f"    type={meta.get('type')!r}  raw_pipe_rows={len(rr)}  "
              + ("CLEAN PIPE TABLE" if len(rr) >= 2 else "no pipe grid in raw"))
        for ln in raw.strip().split("\n"):
            print(f"      | {ln}")
    print()