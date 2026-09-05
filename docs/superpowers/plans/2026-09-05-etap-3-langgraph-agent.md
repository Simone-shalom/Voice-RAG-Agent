# Etap 3 — LangGraph Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph agent that routes queries through search tools with conditional edges — simple questions call one tool and stop; complex questions (e.g. cross-episode comparison) call multiple tools before answering.

**Architecture:** State machine with typed `AgentState` (messages + step_count). Four tools expose the knowledge base: `search_transcripts`, `get_context_around_timestamp`, `compare_across_episodes`, `summarise_segment`. The graph has a single `agent` node (Claude) and a `tools` node (ToolExecutor); conditional edges after `agent` decide whether to call tools or end. `MemorySaver` persists conversation history per `thread_id`. A FastAPI endpoint `/agent/chat` wraps the graph. Integration tests inject a mock LLM to drive deterministic multi-step vs. single-step routing without real API calls.

**Tech Stack:** `langgraph`, `langchain-anthropic`, `langchain-core`, `anthropic`, pytest with mock LLM responses.

---

## File Map

**Create:**
- `backend/app/agent/__init__.py` (empty)
- `backend/app/agent/tools.py` — 4 `@tool`-decorated functions
- `backend/app/agent/graph.py` — LangGraph graph: `build_graph(db, llm)` → `CompiledGraph`
- `backend/app/agent/router.py` — `POST /agent/chat`
- `backend/tests/test_agent_tools.py`
- `backend/tests/test_agent_graph.py`

**Modify:**
- `backend/requirements.txt` — add `langgraph`, `langchain-anthropic`, `langchain-core`
- `backend/app/main.py` — include agent router

---

## Task 1: Add LangGraph dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add dependencies**

Append to `backend/requirements.txt`:
```
langgraph==0.2.50
langchain-anthropic==0.3.3
langchain-core==0.3.25
```

- [ ] **Step 2: Install into venv**

```bash
cd backend && .venv/Scripts/pip.exe install langgraph==0.2.50 langchain-anthropic==0.3.3 langchain-core==0.3.25
```
Expected: `Successfully installed langgraph-...`

- [ ] **Step 3: Verify import**

```bash
cd backend && .venv/Scripts/python.exe -c "import langgraph; print('langgraph', langgraph.__version__)"
```
Expected: `langgraph 0.2.50`

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore(etap-3): add LangGraph and LangChain-Anthropic dependencies"
```

---

## Task 2: Agent tools

**Files:**
- Create: `backend/app/agent/__init__.py`
- Create: `backend/app/agent/tools.py`
- Create: `backend/tests/test_agent_tools.py`

Each tool takes only serializable arguments (strings, floats) — LangGraph serializes tool calls to JSON. DB session is injected via a closure at graph-build time. Tools return plain strings (the LLM reads them as tool output).

- [ ] **Step 1: Write failing tests**

`backend/tests/test_agent_tools.py`:

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.agent.tools import make_tools


def make_db():
    db = MagicMock()
    return db


def test_make_tools_returns_four_tools():
    tools = make_tools(make_db())
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {
        "search_transcripts",
        "get_context_around_timestamp",
        "compare_across_episodes",
        "summarise_segment",
    }


def test_search_transcripts_returns_string():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "hello world", "similarity": 0.9}
    ]):
        tools = make_tools(db)
        search = next(t for t in tools if t.name == "search_transcripts")
        result = search.invoke({"query": "hello", "limit": 5})

    assert isinstance(result, str)
    assert "hello world" in result
    assert "0.0s" in result


def test_get_context_around_timestamp_returns_string():
    db = make_db()
    db.execute.return_value.fetchall.return_value = [
        MagicMock(id="c1", episode_id="ep1", start_ts=4.0, end_ts=9.0, text="nearby chunk")
    ]
    tools = make_tools(db)
    fn = next(t for t in tools if t.name == "get_context_around_timestamp")
    result = fn.invoke({"episode_id": "ep1", "timestamp": 5.0, "window_seconds": 10.0})

    assert isinstance(result, str)
    assert "nearby chunk" in result


def test_compare_across_episodes_returns_string():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "ep1 content", "similarity": 0.9},
        {"chunk_id": "c2", "episode_id": "ep2", "start_ts": 10.0, "end_ts": 15.0,
         "text": "ep2 content", "similarity": 0.8},
    ]):
        tools = make_tools(db)
        fn = next(t for t in tools if t.name == "compare_across_episodes")
        result = fn.invoke({"query": "topic", "episode_ids": "ep1,ep2", "limit_per_episode": 3})

    assert isinstance(result, str)
    assert "ep1" in result
    assert "ep2" in result


def test_summarise_segment_returns_string():
    tools = make_tools(make_db())
    fn = next(t for t in tools if t.name == "summarise_segment")
    result = fn.invoke({"text": "This is a long passage about AI that needs summarising."})

    assert isinstance(result, str)
    # summarise_segment just wraps the text in a prompt template — returns the text formatted
    assert "AI" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.agent'`

- [ ] **Step 3: Create `backend/app/agent/__init__.py`**

Empty file.

- [ ] **Step 4: Create `backend/app/agent/tools.py`**

```python
from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..search.searcher import hybrid_search

MAX_CONTEXT_CHARS = 2000


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No results found."
    lines = []
    for c in chunks:
        ts = f"{c['start_ts']:.1f}s–{c['end_ts']:.1f}s"
        lines.append(f"[{c['episode_id']} @ {ts}] {c['text']}")
    return "\n".join(lines)


def make_tools(db: Session) -> list:
    """
    Build the agent's tool list with db session captured in closure.
    Returns 4 LangChain @tool instances.
    """

    @tool
    def search_transcripts(query: str, limit: int = 5) -> str:
        """Search across all podcast transcripts using hybrid search (semantic + BM25 + RRF).
        Returns the most relevant chunks with timestamps."""
        results = hybrid_search(query=query, limit=limit, db=db)
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
        return "\n".join(f"[{r.start_ts:.1f}s–{r.end_ts:.1f}s] {r.text}" for r in rows)

    @tool
    def compare_across_episodes(query: str, episode_ids: str, limit_per_episode: int = 3) -> str:
        """Search for a topic within specific episodes and return results grouped by episode.
        episode_ids: comma-separated episode UUIDs.
        Use this when the user asks to compare what different episodes say about the same topic."""
        ids = [e.strip() for e in episode_ids.split(",") if e.strip()]
        all_results = hybrid_search(query=query, limit=limit_per_episode * len(ids), db=db)
        by_episode: dict[str, list[dict]] = {}
        for chunk in all_results:
            ep = chunk["episode_id"]
            if ep in ids:
                by_episode.setdefault(ep, []).append(chunk)
        if not by_episode:
            return f"No results for '{query}' in the specified episodes."
        parts = []
        for ep_id in ids:
            chunks = by_episode.get(ep_id, [])
            parts.append(f"=== Episode {ep_id} ===")
            parts.append(_format_chunks(chunks[:limit_per_episode]))
        return "\n".join(parts)

    @tool
    def summarise_segment(text: str) -> str:
        """Format a transcript segment for the assistant to summarise.
        Pass the raw transcript text; returns it formatted for analysis.
        (The LLM is the one that actually summarises — this tool structures the input.)"""
        truncated = text[:MAX_CONTEXT_CHARS]
        return f"[SEGMENT TO SUMMARISE]\n{truncated}"

    return [search_transcripts, get_context_around_timestamp, compare_across_episodes, summarise_segment]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/__init__.py backend/app/agent/tools.py backend/tests/test_agent_tools.py
git commit -m "feat(etap-3): add LangGraph agent tools (search_transcripts, get_context, compare, summarise)"
```

---

## Task 3: LangGraph graph

**Files:**
- Create: `backend/app/agent/graph.py`
- Create: `backend/tests/test_agent_graph.py`

The graph has two nodes: `agent` (LLM with bound tools) and `tools` (ToolNode). Conditional edge after `agent`: if the last message is an `AIMessage` with `tool_calls` → go to `tools`; else → `END`. After `tools` → back to `agent`. Step counter prevents infinite loops — if `step_count >= MAX_STEPS`, force END regardless of tool calls.

`MemorySaver` stores conversation history per `thread_id` so subsequent questions in the same conversation have context.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_agent_graph.py`:

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_graph, MAX_STEPS


def make_db():
    return MagicMock()


def make_mock_llm(responses: list):
    """Create a mock LLM that returns responses in sequence."""
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = responses
    return mock


def test_build_graph_returns_compiled_graph():
    db = make_db()
    mock_llm = make_mock_llm([AIMessage(content="hello")])
    graph = build_graph(db=db, llm=mock_llm)
    assert graph is not None


def test_max_steps_constant_exists():
    assert isinstance(MAX_STEPS, int)
    assert MAX_STEPS >= 3


def test_simple_query_ends_without_tool_call():
    """A direct answer (no tool_calls) ends the graph in one agent step."""
    db = make_db()
    mock_llm = make_mock_llm([
        AIMessage(content="The answer is 42.")  # no tool_calls
    ])
    graph = build_graph(db=db, llm=mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 6 times 7?")], "step_count": 0},
        config={"configurable": {"thread_id": "test-simple"}},
    )

    assert mock_llm.invoke.call_count == 1
    assert result["messages"][-1].content == "The answer is 42."


def test_multi_step_query_calls_tools_then_answers():
    """Query that requires a tool call: agent calls tool, then answers."""
    db = make_db()

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "search_transcripts",
            "args": {"query": "machine learning", "limit": 5},
            "type": "tool_call",
        }],
    )
    final_answer = AIMessage(content="Machine learning was discussed at 12.3s.")

    mock_llm = make_mock_llm([tool_call_msg, final_answer])

    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 12.3, "end_ts": 18.0,
         "text": "machine learning basics", "similarity": 0.9}
    ]):
        graph = build_graph(db=db, llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Where is ML discussed?")], "step_count": 0},
            config={"configurable": {"thread_id": "test-multi"}},
        )

    # LLM was called twice: once for tool call, once for final answer
    assert mock_llm.invoke.call_count == 2
    assert result["messages"][-1].content == "Machine learning was discussed at 12.3s."


def test_step_limit_prevents_infinite_loop():
    """When step_count reaches MAX_STEPS, graph must stop even if LLM wants more tools."""
    db = make_db()

    # LLM always wants to call another tool — should be stopped by step limit
    always_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "id": f"call_x",
            "name": "search_transcripts",
            "args": {"query": "loop"},
            "type": "tool_call",
        }],
    )
    mock_llm = make_mock_llm([always_tool_call] * (MAX_STEPS + 5))

    with patch("app.agent.tools.hybrid_search", return_value=[]):
        graph = build_graph(db=db, llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Loop forever")], "step_count": 0},
            config={"configurable": {"thread_id": "test-limit"}},
        )

    # Should have stopped at MAX_STEPS, not gone forever
    assert mock_llm.invoke.call_count <= MAX_STEPS
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.agent.graph'`

- [ ] **Step 3: Create `backend/app/agent/graph.py`**

```python
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from .tools import make_tools

MAX_STEPS = 6


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def build_graph(db: Session, llm):
    """
    Build and compile the LangGraph agent graph.

    db: SQLAlchemy session (captured in tool closures)
    llm: LangChain chat model (e.g. ChatAnthropic)

    Returns a compiled graph with MemorySaver checkpointing.
    """
    tools = make_tools(db)
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

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/graph.py backend/tests/test_agent_graph.py
git commit -m "feat(etap-3): add LangGraph agent graph with conditional routing and step limit"
```

---

## Task 4: Agent FastAPI router

**Files:**
- Create: `backend/app/agent/router.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/conftest.py` (add agent router mount)
- Create: `backend/tests/test_agent_router.py`

`POST /agent/chat` accepts `{message: str, thread_id: str}` and returns `{reply: str, steps: int}`. The graph is built once per request for simplicity (in production it would be cached per thread). `ANTHROPIC_API_KEY` is required at request time — if missing, returns 400.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_agent_router.py`:

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage


def test_agent_chat_returns_reply(client):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="The answer is 42.")],
        "step_count": 1,
    }

    with patch("app.agent.router.build_graph", return_value=mock_graph):
        response = client.post("/agent/chat", json={"message": "What is 6x7?", "thread_id": "t1"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "The answer is 42."
    assert body["steps"] == 1


def test_agent_chat_missing_message_returns_422(client):
    response = client.post("/agent/chat", json={"thread_id": "t1"})
    assert response.status_code == 422


def test_agent_chat_no_anthropic_key_returns_400(client):
    with patch.dict(os.environ, {}, clear=True), \
         patch("app.agent.router.build_graph", side_effect=RuntimeError("ANTHROPIC_API_KEY not set")):
        response = client.post("/agent/chat", json={"message": "hi", "thread_id": "t2"})
    assert response.status_code == 400
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_router.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.agent.router'`

- [ ] **Step 3: Create `backend/app/agent/router.py`**

```python
import os

from fastapi import APIRouter, Depends, HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .graph import build_graph

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str


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
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Update `backend/app/main.py` to include agent router**

Read current main.py:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .db.session import engine, init_db
from .ingest.router import router as ingest_router
from .search.router import router as search_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield

app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(search_router)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

Add `from .agent.router import router as agent_router` and `app.include_router(agent_router)`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .db.session import engine, init_db
from .ingest.router import router as ingest_router
from .search.router import router as search_router
from .agent.router import router as agent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield

app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_router.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Run full test suite**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/router.py backend/app/main.py backend/tests/test_agent_router.py
git commit -m "feat(etap-3): add POST /agent/chat endpoint with LangGraph routing"
```

---

## Task 5: Integration test — different step counts

**Files:**
- Create: `backend/tests/test_agent_integration.py`

This is the DoD test. It must demonstrate — using mock LLM — that the same graph routes differently for a simple question (1 LLM call, no tool) vs. a complex question (2+ LLM calls, tool used). This test proves the agent is actually a conditional graph, not a fixed chain.

- [ ] **Step 1: Write the integration test**

`backend/tests/test_agent_integration.py`:

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph


def make_db():
    return MagicMock()


def make_sequential_llm(*responses):
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = list(responses)
    return mock


def test_simple_question_uses_one_llm_call():
    """
    A factual question the LLM can answer without tools should result in:
    - 1 LLM call (no tool calls in the response)
    - step_count == 1
    DoD requirement: simple query takes fewer steps than complex query.
    """
    mock_llm = make_sequential_llm(
        AIMessage(content="The capital of France is Paris.")  # direct answer, no tool_calls
    )
    graph = build_graph(db=make_db(), llm=mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")], "step_count": 0},
        config={"configurable": {"thread_id": "integration-simple"}},
    )

    assert mock_llm.invoke.call_count == 1
    assert result["step_count"] == 1
    assert "Paris" in result["messages"][-1].content


def test_multi_hop_question_uses_multiple_llm_calls():
    """
    A question that needs transcript search results in:
    - First LLM call → tool call (search_transcripts)
    - Tool executes → returns results
    - Second LLM call → final answer
    - step_count == 2
    DoD requirement: complex query takes more steps than simple query.
    """
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_abc",
            "name": "search_transcripts",
            "args": {"query": "neural networks introduction", "limit": 5},
            "type": "tool_call",
        }],
    )
    final_response = AIMessage(
        content="Neural networks were introduced at 5.2s in episode ep1."
    )
    mock_llm = make_sequential_llm(tool_call_response, final_response)

    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 5.2, "end_ts": 15.0,
         "text": "an introduction to neural networks", "similarity": 0.95}
    ]):
        graph = build_graph(db=make_db(), llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Where are neural networks discussed?")],
             "step_count": 0},
            config={"configurable": {"thread_id": "integration-multi"}},
        )

    assert mock_llm.invoke.call_count == 2
    assert result["step_count"] == 2
    assert "5.2s" in result["messages"][-1].content


def test_simple_step_count_less_than_multi_hop():
    """Explicit DoD assertion: simple question produces fewer steps."""
    # Simple
    simple_llm = make_sequential_llm(AIMessage(content="Direct answer."))
    simple_graph = build_graph(db=make_db(), llm=simple_llm)
    simple_result = simple_graph.invoke(
        {"messages": [HumanMessage(content="Simple question")], "step_count": 0},
        config={"configurable": {"thread_id": "dod-simple"}},
    )

    # Multi-hop
    multi_llm = make_sequential_llm(
        AIMessage(content="", tool_calls=[{
            "id": "c1", "name": "search_transcripts",
            "args": {"query": "topic"}, "type": "tool_call",
        }]),
        AIMessage(content="Found at 10.0s."),
    )
    with patch("app.agent.tools.hybrid_search", return_value=[]):
        multi_graph = build_graph(db=make_db(), llm=multi_llm)
        multi_result = multi_graph.invoke(
            {"messages": [HumanMessage(content="Complex multi-hop question")], "step_count": 0},
            config={"configurable": {"thread_id": "dod-multi"}},
        )

    assert simple_result["step_count"] < multi_result["step_count"], (
        f"Simple ({simple_result['step_count']}) should be < multi ({multi_result['step_count']})"
    )
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_agent_integration.py -v
```
Expected: 3 passed.

- [ ] **Step 3: Run full test suite**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_agent_integration.py
git commit -m "test(etap-3): add integration tests proving different step counts for simple vs. multi-hop queries"
```

---

## Definition of Done check

- [ ] `POST /agent/chat` accepts `{message, thread_id}` and returns `{reply, steps}`
- [ ] `test_simple_question_uses_one_llm_call`: step_count == 1
- [ ] `test_multi_hop_question_uses_multiple_llm_calls`: step_count == 2
- [ ] `test_simple_step_count_less_than_multi_hop`: explicit comparison assertion
- [ ] `test_step_limit_prevents_infinite_loop`: agent stops at MAX_STEPS
- [ ] All 4 tools tested (search_transcripts, get_context_around_timestamp, compare_across_episodes, summarise_segment)
- [ ] Full test suite green

---

## Self-Review

**Spec coverage:**
- [x] `search_transcripts`, `get_context_around_timestamp`, `compare_across_episodes`, `summarise_segment` → Task 2
- [x] Conditional edges (tool call vs. direct answer) → Task 3 `should_continue`
- [x] `MemorySaver` checkpointing → Task 3 `build_graph`
- [x] Step limit guard → Task 3 `MAX_STEPS`, `test_step_limit_prevents_infinite_loop`
- [x] Different step counts for simple vs. complex → Task 5 integration tests
- [x] FastAPI endpoint → Task 4

**Placeholder scan:** None — all tasks include complete code.

**Type consistency:**
- `make_tools(db) -> list` — matches usage in `build_graph` ✓
- `build_graph(db, llm) -> CompiledGraph` — matches router usage ✓
- `AgentState.step_count` incremented in `agent_node`, checked in `should_continue` ✓
- `ChatResponse.steps = result["step_count"]` ✓
- Tool return type is `str` — LangGraph `ToolNode` handles `str` return values ✓
