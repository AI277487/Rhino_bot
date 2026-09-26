"""
MCP server (thin client) for RhinoBot's ENT retrieval.
Forwards questions to the already-running bot over localhost, so it loads NO
models itself — instant startup, tiny memory. Exposes two tools:
  - search_ent: raw grounded passages (no LLM)
  - ask_ent:    full citation-grounded answer
"""
import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv("/home/botuser/rhino-bot/.env")
BOT_URL = os.environ.get("BOT_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("INTERNAL_API_KEY", "")

mcp = FastMCP("rhino-ent")


def _post(path: str, message: str) -> dict:
    r = httpx.post(
        f"{BOT_URL}{path}",
        headers={"x-internal-key": KEY, "Content-Type": "application/json"},
        json={"message": message},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


@mcp.tool()
def search_ent(question: str) -> str:
    """Retrieve the most relevant ENT textbook passages for a question.

    Hybrid retrieval (BM25 + vector) over four ENT textbooks; returns the top
    passages with their source book and page. No LLM call — fast and cheap.
    """
    data = _post("/internal/search", question)
    passages = data.get("passages", [])
    if not passages:
        return "No relevant passages found in the ENT corpus."
    return "\n\n".join(
        f"[{i}] {p.get('source','unknown')}, p.{p.get('page','?')}\n{p.get('text','').strip()}"
        for i, p in enumerate(passages, 1)
    )


@mcp.tool()
def ask_ent(question: str) -> str:
    """Answer an ENT question, grounded in four ENT textbooks with citations.

    Runs the full RhinoBot pipeline and returns a written answer with
    [book, p.N] citations. Use when you want an answer, not just raw passages.
    """
    data = _post("/internal/chat", question)
    cites = "; ".join(f"{c['book']} p.{c['page']}" for c in data.get("citations", [])) or "none"
    tag = "grounded in textbook sources" if data.get("grounded") else "general ENT knowledge (not citation-grounded)"
    return f"{data.get('answer','')}\n\n---\nCitations: {cites}\nAnswer type: {tag}"


if __name__ == "__main__":
    mcp.run()
