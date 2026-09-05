# Etap 6: MCP Server

## What was added

**`mcp-server/tools.py`** — pure httpx functions (no MCP dependency, testable in backend venv):
- `search_knowledge_base(query, base_url, limit, mode)` → `GET /search?q=...&mode=hybrid`
- `get_episode_summary(episode_id, base_url, chunk_limit)` → `GET /episodes/{id}` + `GET /episodes/{id}/chunks`
- `find_mentions(topic, base_url, limit)` → `GET /search?q=...&mode=hybrid`
- `format_chunks_for_mcp(chunks)` → formatted string with numbered results and timestamps

**`mcp-server/server.py`** — FastMCP server (uses `mcp` package, separate venv):
```python
mcp = FastMCP("Voice Knowledge Agent")
@mcp.tool() def search_knowledge_base_tool(query: str, limit: int = 5) -> str: ...
@mcp.tool() def get_episode_summary_tool(episode_id: str) -> str: ...
@mcp.tool() def find_mentions_tool(topic: str, limit: int = 10) -> str: ...
```

**`backend/app/episodes/router.py`** — new endpoints (registered in main.py):
- `GET /episodes` — list all episodes with chunk counts
- `GET /episodes/{id}` — single episode metadata (404 if not found)
- `GET /episodes/{id}/chunks?limit=N` — first N transcript chunks ordered by start_ts

## Claude Desktop setup

```bash
cd mcp-server
pip install -r requirements.txt   # needs its own venv, conflicts with backend
python server.py                  # runs stdio MCP server
```

Add to `claude_desktop_config.json` (see `mcp-server/claude_desktop_config.json.example`):
```json
{
  "mcpServers": {
    "voice-knowledge": {
      "command": "python",
      "args": ["/path/to/Voice-RAG-Agent/mcp-server/server.py"],
      "env": {"BACKEND_URL": "http://localhost:8000"}
    }
  }
}
```

## Test counts (16 new)
- `test_episodes_router.py` — 5 tests: list, empty, 404, get, chunks
- `test_mcp_tools.py` — 11 tests: search, get_episode, find_mentions, format_chunks

Total backend: **131 tests passing**.

## Key note
MCP package conflicts with fastapi on starlette version — keep in separate venv. Tests for `tools.py` run fine in the backend venv (only httpx needed).
