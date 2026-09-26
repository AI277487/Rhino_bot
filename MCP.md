# RhinoBot MCP Server

Exposes RhinoBot's ENT retrieval as tools any MCP client (Claude Desktop, IDEs)
can call, over the Model Context Protocol.

## Tools
- `search_ent(question)` — hybrid BM25 + vector retrieval; returns grounded
  passages with source book and page. No LLM call.
- `ask_ent(question)`  — full pipeline; returns a citation-grounded answer.

## Architecture — thin client
The RAG engine (MiniLM embeddings + a 94k-chunk BM25 index) is already loaded by
the running FastAPI service. Loading a second copy OOMs a 4 GB box, so the MCP
server is a **thin client**: it forwards questions to an internal localhost
endpoint on the running service instead of loading its own engine.

```
Claude Desktop --stdio over SSH--> mcp_server.py --HTTP localhost--> FastAPI (/internal/*) --> query.py (RAG)
```

- `/internal/search` and `/internal/chat` bypass the app's Google login and are
  guarded by a shared secret (`INTERNAL_API_KEY`, kept in `.env`, never committed).
- Result: the MCP server starts instantly and adds ~no memory.

## Test
```
pip install "mcp[cli]<2" httpx
python test_client.py      # spawns the server, lists tools, calls search_ent
```

## Connect from Claude Desktop (remote, over SSH)
`claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rhino-ent": {
      "command": "ssh",
      "args": ["root@SERVER_IP",
               "/path/to/venv/bin/python /path/to/mcp_server.py"]
    }
  }
}
```
