# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Voice Knowledge Agent's knowledge base as three MCP tools consumable from Claude Desktop and Claude Code, without duplicating backend database logic.

**Architecture:** Standalone `mcp-server/` directory with its own venv. Tool functions live in `tools.py` (pure httpx, no MCP dependency) so they're testable in the backend venv. `server.py` wraps them with FastMCP decorators. Backend gets three new REST endpoints for episode listing/retrieval.

**Tech Stack:** Python 3.12, mcp>=1.0 (FastMCP), httpx, FastAPI (backend additions), pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `mcp-server/tools.py` | `search_knowledge_base`, `get_episode_summary`, `find_mentions`, `format_chunks_for_mcp` — pure httpx |
| `mcp-server/server.py` | `FastMCP("Voice Knowledge Agent")` with three `@mcp.tool()` registrations |
| `mcp-server/requirements.txt` | `mcp>=1.0 httpx>=0.27` |
| `mcp-server/claude_desktop_config.json.example` | Copy-paste config snippet |
| `backend/app/episodes/__init__.py` | Package marker |
| `backend/app/episodes/router.py` | `GET /episodes`, `GET /episodes/{id}`, `GET /episodes/{id}/chunks` |
| `backend/app/main.py` | Include episodes_router |
| `backend/tests/test_episodes_router.py` | 5 tests: list, empty, 404, get, chunks |
| `backend/tests/test_mcp_tools.py` | 11 tests: search, get_episode, find_mentions, format |

---

### Task 1: Backend episodes API

**Files:**
- Create: `backend/app/episodes/router.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_episodes_router.py
def test_list_episodes_returns_list(client, mock_db):
    ep = make_episode_row()
    mock_db.execute.return_value.fetchall.return_value = [ep]
    response = client.get("/episodes")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_get_episode_returns_404_when_missing(client, mock_db):
    mock_db.execute.return_value.fetchone.return_value = None
    response = client.get(f"/episodes/{uuid.uuid4()}")
    assert response.status_code == 404

def test_get_episode_chunks_returns_list(client, mock_db):
    chunks = [make_chunk_row(i*10.0, i*10.0+10.0, f"text {i}") for i in range(3)]
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{uuid.uuid4()}/chunks")
    assert len(response.json()) == 3
```

- [ ] **Step 2: Run — expect FAIL** (route doesn't exist)

- [ ] **Step 3: Implement `backend/app/episodes/router.py`**

```python
router = APIRouter(prefix="/episodes", tags=["episodes"])

class EpisodeSummary(BaseModel):
    id: str
    filename: str
    source_url: str | None
    chunk_count: int

@router.get("", response_model=list[EpisodeSummary])
def list_episodes(db=Depends(get_db)):
    rows = db.execute(text("""
        SELECT e.id, e.filename, e.source_url, COUNT(c.id) AS chunk_count
        FROM episodes e LEFT JOIN chunks c ON c.episode_id = e.id
        GROUP BY e.id, e.filename, e.source_url ORDER BY e.created_at DESC
    """)).fetchall()
    return [EpisodeSummary(id=str(r.id), filename=r.filename,
                           source_url=r.source_url, chunk_count=r.chunk_count) for r in rows]

@router.get("/{episode_id}", response_model=EpisodeSummary)
def get_episode(episode_id: str, db=Depends(get_db)):
    row = db.execute(text("SELECT ... WHERE e.id=:ep_id ..."), {"ep_id": episode_id}).fetchone()
    if not row: raise HTTPException(404, "Episode not found")
    return EpisodeSummary(...)

@router.get("/{episode_id}/chunks", response_model=list[ChunkItem])
def get_episode_chunks(episode_id: str, limit: int = 20, db=Depends(get_db)):
    rows = db.execute(text("SELECT ... FROM chunks WHERE episode_id=:ep_id ORDER BY start_ts LIMIT :limit"),
                      {"ep_id": episode_id, "limit": limit}).fetchall()
    return [ChunkItem(id=str(r.id), start_ts=r.start_ts, end_ts=r.end_ts, text=r.text) for r in rows]
```

- [ ] **Step 4: Mount router in `main.py`**

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add backend/app/episodes/ backend/app/main.py backend/tests/test_episodes_router.py
git commit -m "feat(episodes): add episode listing and chunk retrieval endpoints"
```

---

### Task 2: MCP tool functions

**Files:**
- Create: `mcp-server/tools.py`
- Create: `backend/tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing tests (run in backend venv — no mcp package needed)**

```python
# backend/tests/test_mcp_tools.py
def test_search_knowledge_base_calls_search_endpoint():
    with patch("tools.httpx.get", return_value=make_response(CHUNKS)) as mock_get:
        result = search_knowledge_base("What is RAG?")
    assert "/search" in mock_get.call_args[0][0]

def test_format_chunks_for_mcp_empty():
    assert format_chunks_for_mcp([]) == "No results found."

def test_format_chunks_for_mcp_includes_timestamps():
    output = format_chunks_for_mcp(CHUNKS)
    assert "10.0s" in output
    assert "[1]" in output
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `mcp-server/tools.py`**

```python
def search_knowledge_base(query, base_url="http://localhost:8000", limit=5, mode="hybrid"):
    resp = httpx.get(f"{base_url}/search", params={"q": query, "mode": mode, "limit": limit}, timeout=15.0)
    resp.raise_for_status()
    return resp.json()

def get_episode_summary(episode_id, base_url="http://localhost:8000", chunk_limit=10):
    ep = httpx.get(f"{base_url}/episodes/{episode_id}", timeout=10.0).json()
    chunks = httpx.get(f"{base_url}/episodes/{episode_id}/chunks", params={"limit": chunk_limit}, timeout=10.0).json()
    return {"episode": ep, "chunks": chunks}

def find_mentions(topic, base_url="http://localhost:8000", limit=10):
    resp = httpx.get(f"{base_url}/search", params={"q": topic, "mode": "hybrid", "limit": limit}, timeout=15.0)
    resp.raise_for_status()
    return resp.json()

def format_chunks_for_mcp(chunks):
    if not chunks: return "No results found."
    return "\n\n".join(f"[{i}] Episode {c['episode_id']} @ {c['start_ts']:.1f}s–{c['end_ts']:.1f}s\n{c['text']}"
                       for i, c in enumerate(chunks, 1))
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add mcp-server/tools.py backend/tests/test_mcp_tools.py
git commit -m "feat(mcp): add tool functions with httpx backend calls"
```

---

### Task 3: MCP server registration

**Files:**
- Create: `mcp-server/server.py`
- Create: `mcp-server/requirements.txt`
- Create: `mcp-server/claude_desktop_config.json.example`

- [ ] **Step 1: Create `mcp-server/requirements.txt`**

```
mcp>=1.0
httpx>=0.27
```

> **Warning:** `mcp>=1.0` requires `starlette>=1.0.0` which conflicts with `fastapi 0.115.0` (`starlette<0.39.0`). Use a separate venv for `mcp-server/`.

- [ ] **Step 2: Create `mcp-server/server.py`**

```python
from mcp.server.fastmcp import FastMCP
from tools import search_knowledge_base, get_episode_summary, find_mentions, format_chunks_for_mcp

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
mcp = FastMCP("Voice Knowledge Agent")

@mcp.tool()
def search_knowledge_base_tool(query: str, limit: int = 5) -> str:
    """Search podcast transcripts using hybrid retrieval."""
    return format_chunks_for_mcp(search_knowledge_base(query, BASE_URL, limit))

@mcp.tool()
def get_episode_summary_tool(episode_id: str) -> str:
    """Get metadata and opening transcript for an episode."""
    result = get_episode_summary(episode_id, BASE_URL)
    ep = result["episode"]
    lines = [f"Episode: {ep['filename']}", f"Chunks: {ep['chunk_count']}", "", "Opening:"]
    for c in result["chunks"]:
        lines.append(f"  [{c['start_ts']:.1f}s] {c['text']}")
    return "\n".join(lines)

@mcp.tool()
def find_mentions_tool(topic: str, limit: int = 10) -> str:
    """Find all places in the library where a topic is mentioned."""
    return format_chunks_for_mcp(find_mentions(topic, BASE_URL, limit))

if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: Create `mcp-server/claude_desktop_config.json.example`**

```json
{
  "mcpServers": {
    "voice-knowledge": {
      "command": "python",
      "args": ["/absolute/path/to/Voice-RAG-Agent/mcp-server/server.py"],
      "env": {"BACKEND_URL": "http://localhost:8000"}
    }
  }
}
```

- [ ] **Step 4: Manual smoke test**

```bash
cd mcp-server
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# Start backend: docker compose up backend postgres -d
python server.py  # should print "Voice Knowledge Agent MCP server running"
```

- [ ] **Step 5: Commit**

```bash
git add mcp-server/server.py mcp-server/requirements.txt mcp-server/claude_desktop_config.json.example
git commit -m "feat(mcp): add FastMCP server with three tool registrations"
```

---

### Task 4: Full suite verification + squash

- [ ] **Run full backend suite (131 tests expected)**

```bash
cd backend && python -m pytest tests/ -v
# Expected: 131 passed (115 prior + 5 episode + 11 mcp = 131)
```

- [ ] **Squash-commit to main**

```bash
git reset --soft $(git merge-base HEAD main)
git commit -m "feat(etap-6): add MCP server and episodes API endpoints"
```
