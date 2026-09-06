# Etap 14 — Frontend Correctness & Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the frontend gaps found in a whole-app audit: the project's headline "timestamp citations" feature is built on the backend and rendered by a component, but nothing ever connects the two, so it's invisible in the running app; there's no way to see or choose which search mode powers an answer despite that being a core differentiator; the mic recorder and the chat's audio playback both leak browser resources; and several UI elements are unreachable/unreadable via keyboard or screen reader.

**Architecture:** Backend: the LangGraph agent's tools already retrieve structured chunk data (`{chunk_id, episode_id, start_ts, end_ts, text, similarity}`) but only return it as a formatted string to the LLM — thread an optional `sources_sink` list through `make_tools` → `build_graph` → `stream_agent_response` so the raw chunks are also captured, then emit them as one deduplicated `sources` SSE event before `done`. Also thread an optional `mode` string the same way so the frontend can pick which underlying search function the agent's tools use. Frontend: wire the new SSE event into `ChatUI`'s existing (already-typed but unused) `Message.sources` field, resolve `episode_id` to a filename via a one-time `/api/episodes` fetch, and give `SourcePlayer` a real `<audio>` element using the Media Fragments URI (`#t=start,end`) against the episode's `source_url` when one exists (uploaded files have no persisted audio to play back, so those sources render text-only — there is no server-side audio storage to build real seek-to-clip playback from until a future etap adds one).

**Tech Stack:** FastAPI/LangGraph (backend, unchanged deps), Next.js 15 / React / TypeScript (frontend, unchanged deps).

**Spec:** No separate spec doc — derived from a multi-agent frontend UX/code-quality audit run on 2026-09-05 against `frontend/components/` and `frontend/app/`. Findings referenced inline per task.

## Global Constraints

- Backend: every existing test must keep passing. Run `cd backend && python -m pytest tests/ -v` after Task 1 (expect 206 + new tests, all passing). Follow the existing mocking convention: `patch("app.agent.tools.hybrid_search", ...)` etc. — patch the name where it's imported/used, not at its original definition.
- Frontend: this project has no automated frontend test suite — verify every frontend task by running `docker compose exec frontend sh -c "npx tsc --noEmit"` (must be clean) and confirming the Docker stack is healthy (`docker compose ps`). Windows/Docker does not hot-reload frontend file changes reliably — run `docker compose restart frontend` after edits before checking.
- Do NOT commit after individual tasks. Tasks run sequentially in the current working tree; someone else makes ONE combined commit at the end of the etap (per `CLAUDE.md`: "One squash-commit per etap on `main`"). Do not run `git commit` in any task.
- No `Co-Authored-By` / attribution lines anywhere.
- The non-streaming `POST /agent/chat` endpoint and `graph.invoke` call in `backend/app/agent/router.py::chat` are OUT OF SCOPE for the `mode`/`sources_sink` changes — the frontend only ever calls `/agent/chat/stream`. Leave `chat` untouched; its call to `build_graph(db=db, llm=llm)` keeps working unchanged because the new parameters default to `mode="hybrid"` and `sources_sink=None`.

---

### Task 1: Backend — thread search mode + retrieved-chunk tracking through the agent pipeline, emit an SSE `sources` event

**Finding this fixes:** The frontend's `ChatUI.tsx` already has a typed `Message.sources` field and renders `<SourcePlayer sources={m.sources} />`, but nothing ever sets it — the backend's SSE stream only ever emits `token`/`audio`/`error`/`done`. The project's headline feature (timestamp-cited answers) is dead code. Separately, CLAUDE.md documents four search modes as a core feature, but the agent always uses `hybrid_search` with no way to pick another mode.

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/graph.py`
- Modify: `backend/app/agent/streaming.py`
- Modify: `backend/app/agent/router.py`
- Test: `backend/tests/test_agent_tools.py`
- Test: `backend/tests/test_agent_graph.py`
- Test: `backend/tests/agent/test_streaming.py`
- Test: `backend/tests/agent/test_stream_endpoint.py`

**Interfaces:**
- Produces: `make_tools(db, mode: str = "hybrid", sources_sink: list[dict] | None = None)` — unchanged return shape (4 tools); `build_graph(db, llm, mode: str = "hybrid", sources_sink: list[dict] | None = None)` — unchanged return shape; `stream_agent_response(message, thread_id, db, mode: str = "hybrid")` — same generator, with one new possible SSE event `{"type": "sources", "data": [{"episode_id": str, "start_ts": float, "end_ts": float, "text": str, ...}, ...]}` emitted once, only when at least one chunk was retrieved, right before `{"type": "done"}`.
- `ChatRequest` gains a `mode: Literal["semantic", "bm25", "hybrid", "hybrid+rerank"] = "hybrid"` field, forwarded only by `/agent/chat/stream`.

- [ ] **Step 1: Write the failing tests for `make_tools`'s mode + sources_sink support**

Append to `backend/tests/test_agent_tools.py`:

```python
def test_search_transcripts_appends_to_sources_sink():
    db = make_db()
    sink: list[dict] = []
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "hello world", "similarity": 0.9}
    ]):
        tools = make_tools(db, sources_sink=sink)
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    assert sink == [{"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
                      "text": "hello world", "similarity": 0.9}]


def test_search_transcripts_uses_requested_mode():
    db = make_db()
    with patch("app.agent.tools.semantic_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "semantic hit", "similarity": 0.5}
    ]) as mock_semantic, \
         patch("app.agent.tools.hybrid_search") as mock_hybrid:
        tools = make_tools(db, mode="semantic")
        search = next(t for t in tools if t.name == "search_transcripts")
        result = search.invoke({"query": "hello", "limit": 5})

    mock_semantic.assert_called_once()
    mock_hybrid.assert_not_called()
    assert "semantic hit" in result


def test_unrecognised_mode_falls_back_to_hybrid():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[]) as mock_hybrid:
        tools = make_tools(db, mode="not-a-real-mode")
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    mock_hybrid.assert_called_once()


def test_get_context_around_timestamp_appends_to_sources_sink():
    db = make_db()
    db.execute.return_value.fetchall.return_value = [
        MagicMock(id="c1", episode_id="ep1", start_ts=4.0, end_ts=9.0, text="nearby chunk")
    ]
    sink: list[dict] = []
    tools = make_tools(db, sources_sink=sink)
    fn = next(t for t in tools if t.name == "get_context_around_timestamp")
    fn.invoke({"episode_id": "ep1", "timestamp": 5.0, "window_seconds": 10.0})

    assert sink == [{"episode_id": "ep1", "start_ts": 4.0, "end_ts": 9.0, "text": "nearby chunk"}]


def test_compare_across_episodes_appends_to_sources_sink():
    db = make_db()
    sink: list[dict] = []
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "ep1 content", "similarity": 0.9},
    ]):
        tools = make_tools(db, sources_sink=sink)
        fn = next(t for t in tools if t.name == "compare_across_episodes")
        fn.invoke({"query": "topic", "episode_ids": "ep1", "limit_per_episode": 3})

    assert len(sink) == 1
    assert sink[0]["episode_id"] == "ep1"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_agent_tools.py -v -k "sources_sink or mode"`
Expected: FAIL (`TypeError: make_tools() got an unexpected keyword argument 'sources_sink'`)

- [ ] **Step 3: Implement mode + sources_sink support in `make_tools`**

Replace the full contents of `backend/app/agent/tools.py`:

```python
from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..search.bm25 import bm25_search
from ..search.searcher import hybrid_rerank_search, hybrid_search, semantic_search

MAX_CONTEXT_CHARS = 2000


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No results found."
    lines = []
    for c in chunks:
        ts = f"{c['start_ts']:.1f}s–{c['end_ts']:.1f}s"
        lines.append(f"[{c['episode_id']} @ {ts}] {c['text']}")
    return "\n".join(lines)


def make_tools(db: Session, mode: str = "hybrid", sources_sink: list[dict] | None = None) -> list:
    """
    Build the agent's tool list with db session captured in closure.

    mode: which search function search_transcripts/compare_across_episodes use
      ("semantic" | "bm25" | "hybrid" | "hybrid+rerank"); falls back to hybrid_search
      for an unrecognised value.
    sources_sink: if given, every chunk retrieved by search_transcripts,
      get_context_around_timestamp, or compare_across_episodes is appended to it —
      lets the caller (streaming.py) collect what was actually cited in the answer.

    Returns 4 LangChain @tool instances.
    """
    search_funcs = {
        "semantic": semantic_search,
        "bm25": bm25_search,
        "hybrid": hybrid_search,
        "hybrid+rerank": hybrid_rerank_search,
    }
    search_fn = search_funcs.get(mode, hybrid_search)
    if sources_sink is None:
        sources_sink = []

    @tool
    def search_transcripts(query: str, limit: int = 5) -> str:
        """Search across all podcast transcripts using hybrid search (semantic + BM25 + RRF).
        Returns the most relevant chunks with timestamps."""
        results = search_fn(query=query, limit=limit, db=db)
        sources_sink.extend(results)
        return _format_chunks(results)

    @tool
    def get_context_around_timestamp(episode_id: str, timestamp: float, window_seconds: float = 30.0) -> str:
        """Retrieve transcript chunks from a specific episode near a given timestamp.
        Use this to get more context around a citation you found with search_transcripts."""
        rows = db.execute(
            text(
                """
                SELECT id, episode_id, start_ts, end_ts, text
                FROM chunks
                WHERE episode_id = :ep_id
                  AND start_ts >= :start AND end_ts <= :end
                ORDER BY start_ts
                LIMIT 10
                """
            ),
            {
                "ep_id": episode_id,
                "start": max(0.0, timestamp - window_seconds / 2),
                "end": timestamp + window_seconds / 2,
            },
        ).fetchall()
        if not rows:
            return f"No chunks found in episode {episode_id} near {timestamp:.1f}s."
        for r in rows:
            sources_sink.append({
                "episode_id": str(r.episode_id),
                "start_ts": r.start_ts,
                "end_ts": r.end_ts,
                "text": r.text,
            })
        return "\n".join(f"[{r.start_ts:.1f}s–{r.end_ts:.1f}s] {r.text}" for r in rows)

    @tool
    def compare_across_episodes(query: str, episode_ids: str, limit_per_episode: int = 3) -> str:
        """Search for a topic within specific episodes and return results grouped by episode.
        episode_ids: comma-separated episode UUIDs.
        Use this when the user asks to compare what different episodes say about the same topic."""
        ids = [e.strip() for e in episode_ids.split(",") if e.strip()]
        all_results = search_fn(query=query, limit=limit_per_episode * max(len(ids), 1), db=db)
        by_episode: dict[str, list[dict]] = {}
        for chunk in all_results:
            ep = chunk["episode_id"]
            if ep in ids:
                by_episode.setdefault(ep, []).append(chunk)
        if not by_episode:
            return f"No results for '{query}' in the specified episodes."
        parts = []
        for ep_id in ids:
            chunks = by_episode.get(ep_id, [])[:limit_per_episode]
            sources_sink.extend(chunks)
            parts.append(f"=== Episode {ep_id} ===")
            parts.append(_format_chunks(chunks))
        return "\n".join(parts)

    @tool
    def summarise_segment(text: str) -> str:
        """Format a transcript segment for the assistant to summarise.
        Pass the raw transcript text; returns it formatted for analysis."""
        truncated = text[:MAX_CONTEXT_CHARS]
        return f"[SEGMENT TO SUMMARISE]\n{truncated}"

    return [search_transcripts, get_context_around_timestamp, compare_across_episodes, summarise_segment]
```

- [ ] **Step 4: Run the tools tests to verify they pass, then run the full backend suite**

Run: `cd backend && python -m pytest tests/test_agent_tools.py -v`
Expected: all pass (5 existing + 5 new = 10).

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass (no regressions — `test_compare_across_episodes_returns_string` and the other pre-existing tools tests must still pass unchanged).

- [ ] **Step 5: Write the failing test for `build_graph` forwarding**

Append to `backend/tests/test_agent_graph.py`:

```python
def test_build_graph_forwards_mode_and_sink_to_make_tools():
    mock_llm = make_mock_llm([AIMessage(content="hello")])
    sink: list[dict] = []
    db = make_db()
    with patch("app.agent.graph.make_tools", return_value=[]) as mock_make_tools:
        build_graph(db=db, llm=mock_llm, mode="bm25", sources_sink=sink)

    mock_make_tools.assert_called_once_with(db, mode="bm25", sources_sink=sink)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd backend && python -m pytest tests/test_agent_graph.py -v -k forwards_mode`
Expected: FAIL (`TypeError: build_graph() got an unexpected keyword argument 'mode'`)

- [ ] **Step 7: Add mode/sources_sink parameters to `build_graph`**

Modify `backend/app/agent/graph.py` — only the `build_graph` signature and its `make_tools` call change:

```python
def build_graph(db: Session, llm, mode: str = "hybrid", sources_sink: list[dict] | None = None):
    """
    Build and compile the LangGraph agent graph.

    db: SQLAlchemy session (captured in tool closures)
    llm: LangChain chat model (e.g. ChatAnthropic)
    mode / sources_sink: forwarded to make_tools (see tools.py) so the agent's
      search tools use the requested search mode and record what was retrieved.

    Returns a compiled graph with MemorySaver checkpointing.
    """
    tools = make_tools(db, mode=mode, sources_sink=sources_sink)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    memory = MemorySaver()

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if state["step_count"] >= MAX_STEPS:
            return END
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=memory)
```

(Everything above `build_graph` in the file — imports, `MAX_STEPS`, `AgentState` — is unchanged.)

- [ ] **Step 8: Run graph tests to verify they pass, then the full suite**

Run: `cd backend && python -m pytest tests/test_agent_graph.py -v`
Expected: all pass (5 existing + 1 new = 6).

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 9: Write the failing tests for the SSE `sources` event**

Append to `backend/tests/agent/test_streaming.py`:

```python
@pytest.mark.asyncio
async def test_yields_sources_event_from_sink():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append({"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"})
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    sources_events = [i for i in items if i["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["data"] == [{"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"}]
    # sources event must come before done
    assert items[-1]["type"] == "done"
    assert items[-2]["type"] == "sources"


@pytest.mark.asyncio
async def test_sources_event_deduplicates_identical_chunks():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream
    dupe = {"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"}

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append(dupe)
            sources_sink.append(dict(dupe))
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    sources_events = [i for i in items if i["type"] == "sources"]
    assert len(sources_events[0]["data"]) == 1


@pytest.mark.asyncio
async def test_no_sources_event_when_sink_empty():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    assert not any(i["type"] == "sources" for i in items)


@pytest.mark.asyncio
async def test_forwards_mode_to_build_graph():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph) as mock_build, \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        await _collect(stream_agent_response("hi", "t1", db=MagicMock(), mode="bm25"))

    assert mock_build.call_args.kwargs.get("mode") == "bm25"
```

- [ ] **Step 10: Run them to verify they fail**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v -k "sources or forwards_mode"`
Expected: FAIL (`TypeError: stream_agent_response() got an unexpected keyword argument 'mode'`, and no `sources` events ever produced).

- [ ] **Step 11: Implement `mode` + the `sources` SSE event in `stream_agent_response`**

Replace the full contents of `backend/app/agent/streaming.py`:

```python
# backend/app/agent/streaming.py
import asyncio
import base64
import json
import os
from typing import AsyncGenerator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from .chunker import SentenceChunker
from .graph import build_graph
from ..voice.tts import synthesise


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_agent_response(
    message: str,
    thread_id: str,
    db: Session,
    mode: str = "hybrid",
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE strings.

    Event types: token, audio (b64+text), sources, done, error.
      {"type": "token",  "text": "..."}                          — each streamed token
      {"type": "audio",  "data": "<b64>", "text": "..."}          — synthesised sentence chunk, in playback order
      {"type": "sources", "data": [{"episode_id","start_ts","end_ts","text"}, ...]}
                                                                   — deduplicated chunks the agent actually
                                                                     retrieved, sent once before "done";
                                                                     omitted entirely if nothing was retrieved
      {"type": "done"}                                            — end of stream
      {"type": "error", "text": "..."}                            — fatal error (missing API key, or a failure mid-stream)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield _sse({"type": "error", "text": "ANTHROPIC_API_KEY not set"})
        return

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    sources_sink: list[dict] = []
    graph = build_graph(db=db, llm=llm, mode=mode, sources_sink=sources_sink)
    chunker = SentenceChunker()

    async def _tts_chunk(text: str) -> str:
        loop = asyncio.get_running_loop()
        mp3_bytes = await loop.run_in_executor(None, synthesise, text)
        b64 = base64.b64encode(mp3_bytes).decode()
        return _sse({"type": "audio", "data": b64, "text": text})

    tts_tasks: list[asyncio.Task] = []

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)], "step_count": 0},
            config={"configurable": {"thread_id": thread_id}},
            version="v1",
        ):
            if event["event"] != "on_chat_model_stream":
                continue
            token = event["data"]["chunk"].content
            if not isinstance(token, str) or not token:
                continue

            yield _sse({"type": "token", "text": token})

            for sentence in chunker.push(token):
                tts_tasks.append(asyncio.create_task(_tts_chunk(sentence)))

        for sentence in chunker.flush():
            tts_tasks.append(asyncio.create_task(_tts_chunk(sentence)))

        for task in tts_tasks:
            yield await task

        if sources_sink:
            seen = set()
            deduped = []
            for src in sources_sink:
                key = (src.get("episode_id"), src.get("start_ts"), src.get("end_ts"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(src)
            yield _sse({"type": "sources", "data": deduped})

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})

    finally:
        for task in tts_tasks:
            if not task.done():
                task.cancel()
```

- [ ] **Step 12: Run streaming tests to verify they pass, then the full suite**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v`
Expected: all pass (5 existing + 4 new = 9).

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 13: Write the failing tests for `/agent/chat/stream` forwarding `mode`**

Append to `backend/tests/agent/test_stream_endpoint.py`:

```python
def test_stream_endpoint_forwards_mode(client):
    async def fake_stream(*args, **kwargs):
        yield 'data: {"type": "done"}\n\n'

    with patch("app.agent.router.stream_agent_response", side_effect=fake_stream) as mock_stream:
        client.post(
            "/agent/chat/stream",
            json={"message": "hello", "thread_id": "t1", "mode": "bm25"},
        )

    assert mock_stream.call_args.kwargs.get("mode") == "bm25"


def test_stream_endpoint_defaults_mode_to_hybrid(client):
    async def fake_stream(*args, **kwargs):
        yield 'data: {"type": "done"}\n\n'

    with patch("app.agent.router.stream_agent_response", side_effect=fake_stream) as mock_stream:
        client.post("/agent/chat/stream", json={"message": "hello", "thread_id": "t1"})

    assert mock_stream.call_args.kwargs.get("mode") == "hybrid"
```

- [ ] **Step 14: Run them to verify they fail**

Run: `cd backend && python -m pytest tests/agent/test_stream_endpoint.py -v -k mode`
Expected: FAIL (`422 Unprocessable Entity` or `mode` never forwarded — `ChatRequest` has no `mode` field yet).

- [ ] **Step 15: Add `mode` to `ChatRequest` and forward it in `chat_stream`**

Modify `backend/app/agent/router.py` — add the `Literal` import and the `mode` field, and pass it through in `chat_stream` only (`chat` is unchanged, per Global Constraints):

```python
import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.rate_limit import limiter
from ..db.session import get_db
from .graph import build_graph
from .streaming import stream_agent_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

AgentSearchMode = Literal["semantic", "bm25", "hybrid", "hybrid+rerank"]


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    mode: AgentSearchMode = "hybrid"


class ChatResponse(BaseModel):
    reply: str
    steps: int


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)], "step_count": 0},
            config={"configurable": {"thread_id": request.thread_id}},
        )
        last_msg = result["messages"][-1]
        return ChatResponse(reply=last_msg.content, steps=result["step_count"])
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("agent chat failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=body.message,
            thread_id=body.thread_id,
            db=db,
            mode=body.mode,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
```

(The `chat` function's body is shown in full only so the file is unambiguous — it is byte-for-byte identical to its current Etap-13 state; only `ChatRequest` gaining the `mode` field and `chat_stream`'s body/mode forwarding are new.)

- [ ] **Step 16: Run the stream-endpoint tests, then the full backend suite**

Run: `cd backend && python -m pytest tests/agent/test_stream_endpoint.py -v`
Expected: all pass (2 existing + 2 new = 4).

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass (206 + ~14 new from this task).

---

### Task 2: Frontend — wire the `sources` SSE event into ChatUI, upgrade SourcePlayer, add a search-mode selector

**Finding this fixes:** `Message.sources` is never set (dead citations feature); `SourcePlayer` shows the raw episode UUID instead of a filename and has no audio playback at all; there is no UI for choosing/seeing the search mode despite it being a core project feature.

**Files:**
- Modify: `frontend/components/ChatUI.tsx`
- Modify: `frontend/components/SourcePlayer.tsx`

**Interfaces:**
- Consumes: Task 1's new SSE event shape `{"type": "sources", "data": Source[]}` and the `ChatRequest.mode` field (POST body key `mode`).
- Produces: `SourcePlayer` now takes an `episodeById?: Record<string, Episode>` prop (the `Episode` type is already exported from `frontend/components/FileUploader.tsx` — reuse it, don't redefine it).

- [ ] **Step 1: Replace `frontend/components/SourcePlayer.tsx`**

```tsx
"use client";

import { Episode } from "./FileUploader";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Props {
  sources: Source[];
  episodeById?: Record<string, Episode>;
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SourcePlayer({ sources, episodeById = {} }: Props) {
  if (!sources.length) return null;

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs text-gray-400 uppercase tracking-wider">Sources</p>
      {sources.map((src, i) => {
        const episode = episodeById[src.episode_id];
        const label = episode?.filename ?? src.episode_id;
        const audioSrc = episode?.source_url
          ? `${episode.source_url}#t=${src.start_ts},${src.end_ts}`
          : null;

        return (
          <div
            key={i}
            className="bg-gray-800 rounded-lg p-3 text-sm border border-gray-700"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-indigo-400 font-mono text-xs">
                {formatTime(src.start_ts)} – {formatTime(src.end_ts)}
              </span>
              <span className="text-gray-500 text-xs truncate max-w-xs" title={label}>
                {label}
              </span>
            </div>
            <p className="text-gray-300 line-clamp-3">{src.text}</p>
            {audioSrc ? (
              <audio controls preload="none" src={audioSrc} className="mt-2 w-full h-8">
                Your browser does not support audio playback.
              </audio>
            ) : (
              <p className="mt-2 text-[10px] text-gray-600 italic">
                Audio unavailable for uploaded files.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Replace `frontend/components/ChatUI.tsx`**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import SourcePlayer from "./SourcePlayer";
import { Episode } from "./FileUploader";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  streaming?: boolean;
}

type SearchMode = "semantic" | "bm25" | "hybrid" | "hybrid+rerank";

interface Props {
  selectedEpisode: Episode | null;
}

const SEARCH_MODES: { value: SearchMode; label: string }[] = [
  { value: "semantic", label: "Semantic" },
  { value: "bm25", label: "BM25" },
  { value: "hybrid", label: "Hybrid" },
  { value: "hybrid+rerank", label: "Hybrid + Rerank" },
];

export default function ChatUI({ selectedEpisode }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [episodeById, setEpisodeById] = useState<Record<string, Episode>>({});
  const threadId = useRef(crypto.randomUUID());
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    fetch("/api/episodes")
      .then((res) => (res.ok ? res.json() : []))
      .then((eps: Episode[]) => {
        const map: Record<string, Episode> = {};
        for (const ep of eps) map[ep.id] = ep;
        setEpisodeById(map);
      })
      .catch(() => {
        // sources will fall back to showing the raw episode id
      });
  }, []);

  function getAudioCtx(): AudioContext {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }

  function enqueueAudioChunk(base64: string) {
    audioQueueRef.current = audioQueueRef.current.then(async () => {
      try {
        const ctx = getAudioCtx();
        if (ctx.state === "suspended") await ctx.resume();
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
        await new Promise<void>((resolve) => {
          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(ctx.destination);
          src.onended = () => resolve();
          src.start();
        });
      } catch {
        // audio decode/playback failure is non-fatal — text response still shows
      }
    });
  }

  async function sendMessage(userText: string) {
    if (!userText.trim()) return;
    if (loading) return;
    setInput("");
    setLoading(true);

    // Create/resume the AudioContext synchronously, still inside the
    // triggering user-gesture call stack (before any await below) — browsers
    // may refuse to un-suspend an AudioContext once we've hopped past the
    // gesture window (e.g. after the fetch() call below resolves).
    const gestureCtx = getAudioCtx();
    if (gestureCtx.state === "suspended") {
      gestureCtx.resume().catch(() => {});
    }

    const messagePayload = selectedEpisode
      ? `[Focus on episode: "${selectedEpisode.filename}" (id: ${selectedEpisode.id})] ${userText}`
      : userText;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userText },
      { role: "assistant", text: "", streaming: true },
    ]);

    try {
      const res = await fetch("/api/agent/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messagePayload, thread_id: threadId.current, mode }),
      });
      if (!res.ok || !res.body) throw new Error(await res.text());

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;

      while (!done) {
        const { done: readerDone, value } = await reader.read();
        if (readerDone) break;
        buf += decoder.decode(value, { stream: true });

        const events = buf.split("\n\n");
        buf = events.pop() ?? "";

        for (const raw of events) {
          const line = raw.startsWith("data: ") ? raw.slice(6) : raw;
          if (!line.trim()) continue;

          let event: { type: string; text?: string; data?: string | Source[] };
          try {
            event = JSON.parse(line);
          } catch {
            continue;
          }

          if (event.type === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  text: last.text + ((event.text as string) ?? ""),
                };
              }
              return next;
            });
          } else if (event.type === "audio" && typeof event.data === "string") {
            enqueueAudioChunk(event.data);
          } else if (event.type === "sources" && Array.isArray(event.data)) {
            const sources = event.data;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, sources };
              }
              return next;
            });
          } else if (event.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  text: last.text || `Error: ${event.text}`,
                  streaming: false,
                };
              }
              return next;
            });
            done = true;
          } else if (event.type === "done") {
            done = true;
          }
        }
      }

      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, streaming: false };
        }
        return next;
      });
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, text: `Error: ${err}`, streaming: false };
        } else {
          next.push({ role: "assistant", text: `Error: ${err}` });
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full max-w-2xl mx-auto">
      {/* episode focus chip */}
      {selectedEpisode && (
        <div className="shrink-0 px-4 pt-3">
          <div className="inline-flex items-center gap-1.5 bg-indigo-950/70 border border-indigo-700/60 rounded-full px-3 py-1 text-xs text-indigo-300">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
            <span className="truncate max-w-[240px]" title={selectedEpisode.filename}>
              Focused on: <span className="font-medium text-indigo-200">{selectedEpisode.filename}</span>
            </span>
            <span className="text-indigo-500 ml-0.5">·</span>
            <span className="text-indigo-500 text-[10px]">
              deselect in sidebar
            </span>
          </div>
        </div>
      )}

      {/* search mode selector */}
      <div className="shrink-0 px-4 pt-3 flex items-center gap-2">
        <label htmlFor="search-mode" className="text-[10px] uppercase tracking-widest text-gray-500">
          Search mode
        </label>
        <select
          id="search-mode"
          value={mode}
          onChange={(e) => setMode(e.target.value as SearchMode)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-xs text-gray-300 focus:outline-none focus:border-indigo-500"
        >
          {SEARCH_MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-gray-600 text-center max-w-xs">
              {selectedEpisode
                ? `Focused on "${selectedEpisode.filename}". Ask anything about it.`
                : "Ask a question about your audio library, or select an episode in the sidebar to focus on it."}
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-prose rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">
                {m.text}
                {m.streaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle" />
                )}
              </p>
              {m.sources && <SourcePlayer sources={m.sources} episodeById={episodeById} />}
            </div>
          </div>
        ))}
      </div>

      {/* input bar */}
      <div className="shrink-0 border-t border-gray-800 p-4 space-y-3">
        <AudioRecorder onTranscript={(t) => sendMessage(t)} />
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder={
              selectedEpisode
                ? `Ask about "${selectedEpisode.filename}"…`
                : "Type a question…"
            }
            disabled={loading}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-xl text-sm font-semibold"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

Note: this step deliberately leaves the AudioContext-leak-on-unmount fix and all accessibility attributes for later tasks (Task 4) — it is scoped to sources + mode only, to keep this diff reviewable against a single concern.

- [ ] **Step 3: Restart the frontend container and type-check**

Run: `docker compose restart frontend`
Run: `docker compose exec frontend sh -c "npx tsc --noEmit"`
Expected: no type errors.

- [ ] **Step 4: Manually verify in the browser**

With the Docker stack up and real API keys configured (or, if keys aren't yet set, at minimum confirm the UI renders without a console error): open `http://localhost:3000`, confirm the search-mode dropdown appears above the message list and can be changed. If a backend with real keys is available, ask a question that requires a lookup, and confirm a "Sources" section appears under the assistant's reply once the answer finishes streaming. Report exactly what you observed (including if real keys aren't available and this step could only be partially verified via `tsc` + no-console-errors).

---

### Task 3: Frontend — fix the AudioRecorder mic/stream leak on unmount, add accessibility to its own controls

**Finding this fixes:** `AudioRecorder`'s unmount cleanup only calls `stopVadLoop()` (tears down the VAD's `AudioContext`/analyser) but never stops the `MediaRecorder` or the raw `MediaStream` — if the component unmounts while `recording === true`, the microphone stays open indefinitely with the browser's recording indicator still on. Separately, the record button has no `aria-pressed`/`aria-label` (state is conveyed by color alone) and no visible `:focus-visible` styling.

**Files:**
- Modify: `frontend/components/AudioRecorder.tsx`

- [ ] **Step 1: Replace `frontend/components/AudioRecorder.tsx`**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

const SILENCE_THRESHOLD = 0.015; // RMS amplitude below this counts as silence
const SILENCE_TIMEOUT_MS = 1500; // auto-stop after this much continuous silence once speech has started
const MIN_SPEECH_MS = 300; // ignore silence detection until at least this much speech has been heard

export default function AudioRecorder({ onTranscript }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vadStatus, setVadStatus] = useState<"idle" | "listening" | "silence">("idle");
  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechStartedAtRef = useRef<number | null>(null);

  function stopVadLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    audioCtxRef.current = null;
    analyserRef.current = null;
    silenceStartRef.current = null;
    speechStartedAtRef.current = null;
  }

  function monitorSilence() {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - 128) / 128;
      sumSquares += normalized * normalized;
    }
    const rms = Math.sqrt(sumSquares / data.length);

    const now = performance.now();
    if (speechStartedAtRef.current === null && rms > SILENCE_THRESHOLD) {
      speechStartedAtRef.current = now;
    }
    const speechElapsed = speechStartedAtRef.current !== null ? now - speechStartedAtRef.current : 0;

    if (rms > SILENCE_THRESHOLD) {
      silenceStartRef.current = null;
      setVadStatus("listening");
    } else if (speechStartedAtRef.current !== null && speechElapsed > MIN_SPEECH_MS) {
      if (silenceStartRef.current === null) {
        silenceStartRef.current = now;
      }
      setVadStatus("silence");
      if (now - silenceStartRef.current > SILENCE_TIMEOUT_MS) {
        stop();
        return;
      }
    }

    rafRef.current = requestAnimationFrame(monitorSilence);
  }

  async function start() {
    if (recording || mediaRef.current) return; // guard against double-start (rapid clicks) leaking a stream/context
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stopVadLoop();
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        try {
          const res = await fetch("/api/voice/stt", { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          const { text } = await res.json();
          onTranscript(text);
        } catch (err) {
          setError(String(err));
        }
        setVadStatus("idle");
        mediaRef.current = null;
      };
      mr.start();
      mediaRef.current = mr;
      streamRef.current = stream;
      setRecording(true);
      setVadStatus("listening");

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      silenceStartRef.current = null;
      speechStartedAtRef.current = null;
      rafRef.current = requestAnimationFrame(monitorSilence);
    } catch (err) {
      setError(String(err));
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  useEffect(() => {
    return () => {
      stopVadLoop();
      // If the component unmounts mid-recording, there's no point uploading
      // an in-progress clip — detach the normal onstop upload handler before
      // stopping so it doesn't fire, then release the mic directly.
      if (mediaRef.current && mediaRef.current.state !== "inactive") {
        mediaRef.current.onstop = null;
        mediaRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={recording ? stop : start}
        aria-pressed={recording}
        aria-label={recording ? "Stop voice recording" : "Start voice recording"}
        className={`px-6 py-3 rounded-full font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-400 ${
          recording
            ? "bg-red-600 hover:bg-red-700 animate-pulse"
            : "bg-indigo-600 hover:bg-indigo-700"
        }`}
      >
        {recording ? "Stop Recording" : "Ask with Voice"}
      </button>
      {recording && (
        <p className="text-xs text-gray-500" role="status">
          {vadStatus === "silence" ? "Silence detected — stopping soon…" : "Listening…"}
        </p>
      )}
      {error && (
        <p className="text-red-400 text-sm" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Restart the frontend container and type-check**

Run: `docker compose restart frontend`
Run: `docker compose exec frontend sh -c "npx tsc --noEmit"`
Expected: no type errors.

- [ ] **Step 3: Manually verify the leak fix**

With the app open in a browser and dev tools' recording indicator visible: click "Ask with Voice" to start recording, then navigate away in a way that unmounts `ChatUI`/`AudioRecorder` (e.g. if there's no other route, simulate by triggering React Strict Mode's double-invoke in dev, or note this is a code-review-verifiable fix if there's no easy way to force an unmount in this app's current single-page layout). At minimum, confirm clicking the button starts/stops recording exactly as before (no regression), and that `stop()`'s normal flow (clicking "Stop Recording") still uploads and transcribes correctly.

---

### Task 4: Frontend — friendly error messages, FileUploader state fixes, remaining accessibility

**Findings this fixes:** Raw `Error` objects and raw response bodies (potentially JSON/HTML/stack traces) are shown directly in the chat and upload UI; the upload status banner persists across tab switches (a "done" message from the File tab still shows after switching to the RSS tab); the episode list briefly shows "No audio ingested yet." on every load even when episodes exist, because the empty check doesn't distinguish "not loaded yet" from "loaded and genuinely empty"; decorative emoji are read aloud by screen readers; toggle-style buttons (tabs, episode selection) have no `aria-pressed`.

**Files:**
- Create: `frontend/lib/errors.ts`
- Modify: `frontend/components/ChatUI.tsx`
- Modify: `frontend/components/FileUploader.tsx`
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Create the shared error-parsing helper**

Create `frontend/lib/errors.ts`:

```typescript
/**
 * Extracts a clean, user-facing message from a failed fetch Response.
 * Backend errors are JSON `{"detail": "..."}` (see CLAUDE.md's error-handling
 * convention) — fall back to a generic status-code message for anything else
 * (an HTML error page, a stack trace, an empty body) so raw internals never
 * reach the UI.
 */
export async function friendlyErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // not JSON — fall through to a generic message
  }
  return `Request failed (${res.status})`;
}
```

- [ ] **Step 2: Use it in `ChatUI.tsx`, and stop double-prefixing "Error:"**

In `frontend/components/ChatUI.tsx`, add the import at the top:

```tsx
import { friendlyErrorMessage } from "@/lib/errors";
```

Change the line `if (!res.ok || !res.body) throw new Error(await res.text());` to:

```tsx
if (!res.ok || !res.body) throw new Error(await friendlyErrorMessage(res));
```

Change the `catch (err)` block at the end of `sendMessage` from:

```tsx
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, text: `Error: ${err}`, streaming: false };
        } else {
          next.push({ role: "assistant", text: `Error: ${err}` });
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
```

to:

```tsx
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, text: last.text || `Error: ${message}`, streaming: false };
        } else {
          next.push({ role: "assistant", text: `Error: ${message}` });
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
```

(`err instanceof Error` is true here because the only thing thrown in this function is `new Error(...)`; `err.message` avoids the `${err}` string coercion that previously produced a doubled "Error: Error: ..." prefix, since `Error.prototype.toString()` already includes "Error: ".)

- [ ] **Step 3: Add the AudioContext-close-on-unmount fix and per-message `aria-live` to `ChatUI.tsx`**

Add a second `useEffect` (alongside the existing episode-fetch effect — this file already has one `useEffect` for that from Task 2) that closes the playback `AudioContext` when the component unmounts:

```tsx
  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, []);
```

Place it directly after the existing `useEffect` that fetches `/api/episodes`.

Then change the per-message rendering. Screen readers should not re-announce a token-by-token streaming message on every token (that would spam-read the growing sentence); mark a message's live region `"off"` while it is still streaming and only make it `"polite"` once it settles, so the assistive tech announces the complete answer once, not each fragment. Replace:

```tsx
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-prose rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">
                {m.text}
                {m.streaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle" />
                )}
              </p>
              {m.sources && <SourcePlayer sources={m.sources} episodeById={episodeById} />}
            </div>
          </div>
        ))}
```

with:

```tsx
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              aria-live={m.role === "assistant" && m.streaming ? "off" : "polite"}
              aria-atomic="true"
              className={`max-w-prose rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">
                {m.text}
                {m.streaming && (
                  <span
                    aria-hidden="true"
                    className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle"
                  />
                )}
              </p>
              {m.sources && <SourcePlayer sources={m.sources} episodeById={episodeById} />}
            </div>
          </div>
        ))}
```

Finally, add a focus-visible ring to the Send button — change:

```tsx
          <button
            onClick={() => sendMessage(input)}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-xl text-sm font-semibold"
          >
            Send
          </button>
```

to:

```tsx
          <button
            onClick={() => sendMessage(input)}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-xl text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-400"
          >
            Send
          </button>
```

- [ ] **Step 4: Fix `FileUploader.tsx` — friendly errors, per-tab status reset, loading-state-aware empty message, accessibility**

Add the import at the top of `frontend/components/FileUploader.tsx`:

```tsx
import { friendlyErrorMessage } from "@/lib/errors";
```

Add a `loaded` state next to the existing `episodes` state:

```tsx
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [loaded, setLoaded] = useState(false);
```

Update `loadEpisodes` to set it once the fetch settles (success or failure):

```tsx
  async function loadEpisodes() {
    try {
      const res = await fetch("/api/episodes");
      if (res.ok) setEpisodes(await res.json());
    } catch {
      // backend may not be ready on first render
    } finally {
      setLoaded(true);
    }
  }
```

Replace every `throw new Error(await res.text());` (there are three: in `uploadFile`, `ingestUrl`, `ingestRss`) with `throw new Error(await friendlyErrorMessage(res));`, and every `setStatus({ kind: "error", message: String(err) })` with:

```tsx
setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
```

(there are three call sites — one at the end of each of `uploadFile`, `ingestUrl`, `ingestRss` — apply the same substitution to each).

Add a tab-switch helper that resets the status banner, right after the `toggleEpisode` function:

```tsx
  function selectTab(next: "file" | "url" | "rss") {
    setStatus({ kind: "idle" });
    setTab(next);
  }
```

Change the three tab buttons from `onClick={() => setTab("file")}` / `"url"` / `"rss"` to `onClick={() => selectTab("file")}` / `"url"` / `"rss"` respectively, and add `aria-pressed={tab === "file"}` (etc., matching each button's tab value) to each of the three tab `<button>` elements.

Add `aria-hidden="true"` to the decorative music-note emoji:

```tsx
            <span aria-hidden="true" className="text-xl block mb-1">🎵</span>
```

Add `aria-hidden="true"` to the two checkmark spans (the "done" status indicator and the "focused" episode badge) — change:

```tsx
        {status.kind === "done" && (
          <p className="text-xs text-green-400 flex items-center gap-1">
            <span>✓</span>
            <span>
```

to:

```tsx
        {status.kind === "done" && (
          <p className="text-xs text-green-400 flex items-center gap-1">
            <span aria-hidden="true">✓</span>
            <span>
```

and change:

```tsx
                  {selected && (
                    <span className="shrink-0 text-indigo-400 text-[10px] mt-0.5">✓ focused</span>
                  )}
```

to:

```tsx
                  {selected && (
                    <span className="shrink-0 text-indigo-400 text-[10px] mt-0.5">
                      <span aria-hidden="true">✓</span> focused
                    </span>
                  )}
```

Add `aria-pressed={selected}` to the per-episode `<button>` in the episode list (the one with `onClick={() => toggleEpisode(ep)}`).

Finally, replace the loading-unaware empty-state check:

```tsx
        {episodes.length === 0 ? (
          <p className="text-[11px] text-gray-600 text-center py-2">
            No audio ingested yet.
          </p>
        ) : (
```

with:

```tsx
        {!loaded ? (
          <p className="text-[11px] text-gray-600 text-center py-2">Loading…</p>
        ) : episodes.length === 0 ? (
          <p className="text-[11px] text-gray-600 text-center py-2">
            No audio ingested yet.
          </p>
        ) : (
```

(the existing closing `)}` for this ternary does not need to change — you are only adding one more branch in front of the existing two.)

- [ ] **Step 5: Hide the decorative header emoji in `page.tsx`**

In `frontend/app/page.tsx`, change:

```tsx
          <span className="text-xl leading-none">🎙️</span>
```

to:

```tsx
          <span aria-hidden="true" className="text-xl leading-none">🎙️</span>
```

- [ ] **Step 6: Restart the frontend container and type-check**

Run: `docker compose restart frontend`
Run: `docker compose exec frontend sh -c "npx tsc --noEmit"`
Expected: no type errors.

- [ ] **Step 7: Manually verify in the browser**

Open `http://localhost:3000`. Switch between the File/URL/RSS tabs after triggering a status message on one tab (e.g. a failed upload) and confirm the status banner clears when switching tabs. Reload the page and confirm the sidebar briefly shows "Loading…" rather than flashing "No audio ingested yet." before the real list appears (this may be too fast to see on localhost — check the Network tab timing or temporarily throttle to confirm if needed). Confirm the tab buttons and episode list items are keyboard-focusable (Tab key) and show a visible focus outline. Report exactly what you observed.

- [ ] **Step 8: Verify the whole Docker stack is healthy**

Run: `docker compose ps`
Expected: `postgres`, `backend`, `frontend` all `Up`/healthy.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the dead-citations backend gap and the missing search-mode plumbing. Task 2 covers the frontend half of citations (rendering, filename resolution, playback where possible) and the mode selector UI. Task 3 covers the AudioRecorder mic/stream leak and its own accessibility. Task 4 covers the ChatUI AudioContext leak, raw-error disclosure in both ChatUI and FileUploader, the per-tab status bug, the loading-state flash, and the remaining `aria-hidden`/`aria-pressed`/focus-visible gaps. Deliberately deferred (not in this etap): real persistent audio storage for uploaded files (there is currently no server-side audio retention after transcription — building that is a bigger, separate feature), a full `role="tablist"`/`role="tab"` ARIA pattern for the upload tabs (a lighter `aria-pressed` toggle-group pattern was used instead, a legitimate simpler alternative), responsive/mobile layout, dark/light theme toggle, and SSE event runtime-type validation via a discriminated union — all lower-severity findings from the audit, left as backlog.
- **Placeholder scan:** no TBD/TODO markers; every step has literal code.
- **Type/name consistency:** `Source` interface is defined identically in `ChatUI.tsx` and `SourcePlayer.tsx` (both already existed before this plan; unchanged). `episodeById`/`Episode` naming matches `FileUploader.tsx`'s existing exported `Episode` type exactly — no redefinition. `mode`/`sources_sink` parameter names and defaults are identical across `make_tools`, `build_graph`, and `stream_agent_response` (Task 1). `friendlyErrorMessage` is imported with the same name and path (`@/lib/errors`) in both `ChatUI.tsx` and `FileUploader.tsx` (Task 4).
