# Etap 9 — Streaming Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-round-trip chat+TTS flow with a single streaming SSE endpoint that emits LLM tokens in real time, synthesises audio sentence-by-sentence, and plays chunks gaplessly in the browser using AudioContext.

**Architecture:** The backend adds `POST /agent/chat/stream` which uses LangGraph's `astream_events` API to stream tokens, buffers them into sentence chunks at `[.!?]+\s` boundaries (min 8 words), calls ElevenLabs TTS per chunk, and yields each chunk as a base64-encoded SSE `audio` event. The frontend replaces the blocking `fetch` with a streaming reader that renders tokens progressively, decodes incoming base64 MP3 frames via the Web Audio API, and queues AudioBuffers for gap-free playback.

**Tech Stack:** FastAPI `StreamingResponse` + `asyncio`, LangGraph `astream_events(version="v1")`, ElevenLabs TTS (existing `synthesise` fn), Web Audio API `AudioContext`, base64 encoding, SSE format (`data: ...\n\n`).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/agent/chunker.py` | Create | Sentence-boundary chunker (pure fn, no I/O) |
| `backend/app/agent/streaming.py` | Create | `stream_agent_response` async generator |
| `backend/app/agent/router.py` | Modify | Add `POST /agent/chat/stream` SSE endpoint |
| `backend/tests/agent/test_chunker.py` | Create | Unit tests for sentence chunker |
| `backend/tests/agent/test_streaming.py` | Create | Unit tests for streaming generator (mocked graph + TTS) |
| `frontend/components/ChatUI.tsx` | Modify | SSE reader, token streaming, AudioContext playback |

---

### Task 1: Sentence chunker (pure function, fully unit-tested)

**Files:**
- Create: `backend/app/agent/chunker.py`
- Create: `backend/tests/agent/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/agent/test_chunker.py
import pytest
from app.agent.chunker import flush_chunks


def test_returns_empty_on_short_buffer():
    # fewer than 8 words → no flush
    chunks = list(flush_chunks("Hello world."))
    assert chunks == []


def test_flushes_on_sentence_boundary():
    text = "The quick brown fox jumps over the lazy dog. Next sentence begins here now."
    chunks = list(flush_chunks(text))
    assert len(chunks) == 1
    assert chunks[0].startswith("The quick brown fox")


def test_multiple_sentences():
    text = (
        "First sentence is long enough to flush right now. "
        "Second sentence is also long enough to flush as well. "
        "Third sentence completes this longer test input."
    )
    chunks = list(flush_chunks(text))
    assert len(chunks) == 2  # last fragment kept as remainder


def test_remainder_returned_on_none_input():
    # passing None signals end-of-stream — flush everything
    text = "Short but final."
    chunks = list(flush_chunks(text, final=True))
    assert len(chunks) == 1
    assert "Short but final" in chunks[0]


def test_no_flush_below_word_threshold():
    # 6 words, sentence boundary present — still below MIN_WORDS
    chunks = list(flush_chunks("One two three four five."))
    assert chunks == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd backend
python -m pytest tests/agent/test_chunker.py -v
```

Expected: ImportError or failures — `app.agent.chunker` doesn't exist yet.

- [ ] **Step 3: Write the minimal chunker**

```python
# backend/app/agent/chunker.py
import re
from typing import Generator

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 8
_buffer: str = ""


def flush_chunks(text: str, final: bool = False) -> Generator[str, None, None]:
    """
    Stateless: splits `text` at sentence boundaries if the left fragment
    has at least MIN_WORDS words. Pass `final=True` to flush any remainder.

    NOTE: This is a pure function that operates on a single text block.
    Callers maintain the running buffer themselves.
    """
    parts = _BOUNDARY.split(text)
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if not is_last:
            if len(part.split()) >= MIN_WORDS:
                yield part.strip()
        else:
            if final and part.strip():
                yield part.strip()
```

Wait — this is stateless but the real use case needs *accumulation* across tokens. Redesign as a stateful class to be testable and practical:

```python
# backend/app/agent/chunker.py
import re
from typing import Generator

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 8


class SentenceChunker:
    """
    Accumulates streamed tokens and yields sentence-boundary chunks
    of at least MIN_WORDS words.

    Usage:
        chunker = SentenceChunker()
        for token in token_stream:
            for chunk in chunker.push(token):
                synthesise(chunk)
        for chunk in chunker.flush():
            synthesise(chunk)
    """

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> Generator[str, None, None]:
        self._buf += token
        parts = _BOUNDARY.split(self._buf, maxsplit=1)
        while len(parts) == 2:
            left, rest = parts
            if len(left.split()) >= MIN_WORDS:
                yield left.strip()
                self._buf = rest
                parts = _BOUNDARY.split(self._buf, maxsplit=1)
            else:
                break

    def flush(self) -> Generator[str, None, None]:
        remainder = self._buf.strip()
        if remainder:
            yield remainder
            self._buf = ""
```

Update the test file to match the class-based API:

```python
# backend/tests/agent/test_chunker.py
import pytest
from app.agent.chunker import SentenceChunker


def test_no_yield_below_word_threshold():
    c = SentenceChunker()
    result = list(c.push("Hello world."))
    assert result == []


def test_yields_on_sentence_boundary_with_enough_words():
    c = SentenceChunker()
    sentence = "The quick brown fox jumps over the lazy dog. "
    result = list(c.push(sentence))
    assert len(result) == 1
    assert result[0] == "The quick brown fox jumps over the lazy dog."


def test_remainder_stays_in_buffer():
    c = SentenceChunker()
    list(c.push("The quick brown fox jumps over the lazy dog. "))
    # next sentence not yet complete
    result = list(c.push("Incomplete"))
    assert result == []


def test_flush_returns_remainder():
    c = SentenceChunker()
    list(c.push("Full first sentence with enough words here. "))
    list(c.push("Short."))
    remainder = list(c.flush())
    assert len(remainder) == 1
    assert "Short" in remainder[0]


def test_multiple_sentences_in_one_push():
    c = SentenceChunker()
    text = (
        "First long sentence is complete and has enough words right here. "
        "Second long sentence is also complete and has enough words too. "
        "Third partial"
    )
    result = list(c.push(text))
    assert len(result) == 2
    remainder = list(c.flush())
    assert len(remainder) == 1
    assert "Third partial" in remainder[0]


def test_flush_on_empty_buffer():
    c = SentenceChunker()
    assert list(c.flush()) == []
```

- [ ] **Step 4: Run tests — should pass now**

```
python -m pytest tests/agent/test_chunker.py -v
```

Expected: 6/6 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/chunker.py backend/tests/agent/test_chunker.py
git commit -m "feat(agent): add SentenceChunker for streaming TTS boundary detection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AKFBDJAomY3MtdCnku8kQ6"
```

---

### Task 2: `stream_agent_response` async generator

**Files:**
- Create: `backend/app/agent/streaming.py`
- Create: `backend/tests/agent/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/agent/test_streaming.py
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.streaming import stream_agent_response


def _make_token_event(text: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": MagicMock(content=text)},
    }


async def _collect(gen) -> list[dict]:
    results = []
    async for item in gen:
        results.append(json.loads(item.removeprefix("data: ").strip()))
    return results


@pytest.mark.asyncio
async def test_yields_token_events():
    async def fake_stream(*args, **kwargs):
        for word in ["Hello ", "world "]:
            yield _make_token_event(word)

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"mp3bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )

    token_events = [i for i in items if i["type"] == "token"]
    assert len(token_events) >= 1


@pytest.mark.asyncio
async def test_yields_done_event():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )

    assert items[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_audio_event_base64_encoded():
    long_sentence = (
        "The quick brown fox jumps over the lazy dog and then runs away. "
    )

    async def fake_stream(*args, **kwargs):
        yield _make_token_event(long_sentence)

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream
    fake_mp3 = b"\xff\xf3raw_mp3_data"

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=fake_mp3), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )

    audio_events = [i for i in items if i["type"] == "audio"]
    assert len(audio_events) >= 1
    decoded = base64.b64decode(audio_events[0]["data"])
    assert decoded == fake_mp3
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/agent/test_streaming.py -v
```

Expected: ImportError — `app.agent.streaming` doesn't exist yet.

- [ ] **Step 3: Implement `stream_agent_response`**

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
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE strings.

    Event types:
      {"type": "token",  "text": "..."}          — each streamed token
      {"type": "audio",  "data": "<b64>", "text": "..."} — synthesised sentence chunk
      {"type": "done"}                             — end of stream
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield _sse({"type": "error", "text": "ANTHROPIC_API_KEY not set"})
        return

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    graph = build_graph(db=db, llm=llm)
    chunker = SentenceChunker()

    async def _tts_chunk(text: str) -> str:
        """Run blocking TTS in thread pool, return SSE audio event string."""
        loop = asyncio.get_event_loop()
        mp3_bytes = await loop.run_in_executor(None, synthesise, text)
        b64 = base64.b64encode(mp3_bytes).decode()
        return _sse({"type": "audio", "data": b64, "text": text})

    tts_tasks: list[asyncio.Task] = []

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=message)], "step_count": 0},
        config={"configurable": {"thread_id": thread_id}},
        version="v1",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        token: str = event["data"]["chunk"].content
        if not isinstance(token, str) or not token:
            continue

        yield _sse({"type": "token", "text": token})

        for sentence in chunker.push(token):
            task = asyncio.create_task(_tts_chunk(sentence))
            tts_tasks.append(task)

    # flush remaining partial sentence
    for sentence in chunker.flush():
        task = asyncio.create_task(_tts_chunk(sentence))
        tts_tasks.append(task)

    # yield audio events in order (preserves playback sequence)
    for task in tts_tasks:
        yield await task

    yield _sse({"type": "done"})
```

- [ ] **Step 4: Install pytest-asyncio (if not present)**

Check `backend/requirements.txt` — if `pytest-asyncio` isn't listed, add it:

```
pytest-asyncio>=0.23.0
```

Add `asyncio_mode = "auto"` to `backend/pytest.ini` (or `pyproject.toml` `[tool.pytest.ini_options]`):

```ini
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 5: Run tests — should pass**

```
python -m pytest tests/agent/test_streaming.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 6: Run full test suite to check for regressions**

```
python -m pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/streaming.py backend/tests/agent/test_streaming.py
git commit -m "feat(agent): add stream_agent_response async generator with sentence-TTS pipeline

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AKFBDJAomY3MtdCnku8kQ6"
```

---

### Task 3: `POST /agent/chat/stream` FastAPI SSE endpoint

**Files:**
- Modify: `backend/app/agent/router.py`
- Create: `backend/tests/agent/test_stream_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agent/test_stream_endpoint.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import get_db


async def _fake_stream(*args, **kwargs):
    yield 'data: {"type": "token", "text": "Hello"}\n\n'
    yield 'data: {"type": "done"}\n\n'


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_stream_endpoint_returns_200(client):
    with patch("app.agent.router.stream_agent_response", side_effect=_fake_stream), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        res = client.post(
            "/agent/chat/stream",
            json={"message": "hello", "thread_id": "t1"},
        )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]


def test_stream_endpoint_body_contains_done(client):
    with patch("app.agent.router.stream_agent_response", side_effect=_fake_stream), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        res = client.post(
            "/agent/chat/stream",
            json={"message": "hello", "thread_id": "t1"},
        )
    assert b'"type": "done"' in res.content
```

- [ ] **Step 2: Run test to confirm failure**

```
python -m pytest tests/agent/test_stream_endpoint.py -v
```

Expected: 404 — endpoint doesn't exist yet.

- [ ] **Step 3: Add the streaming endpoint to router.py**

Open `backend/app/agent/router.py` and add below the existing `chat` endpoint:

```python
from fastapi.responses import StreamingResponse

from .streaming import stream_agent_response


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=request.message,
            thread_id=request.thread_id,
            db=db,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
```

Full updated `router.py`:

```python
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .graph import build_graph
from .streaming import stream_agent_response

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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=request.message,
            thread_id=request.thread_id,
            db=db,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests — should pass**

```
python -m pytest tests/agent/test_stream_endpoint.py -v
```

Expected: 2/2 PASS.

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/router.py backend/tests/agent/test_stream_endpoint.py
git commit -m "feat(agent): add POST /agent/chat/stream SSE endpoint

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AKFBDJAomY3MtdCnku8kQ6"
```

---

### Task 4: Frontend — AudioContext streaming in ChatUI

**Files:**
- Modify: `frontend/components/ChatUI.tsx`

No automated tests for this task — browser audio APIs can't run in Jest/Node. The verification is manual: start the app and confirm audio plays chunk-by-chunk as text appears.

- [ ] **Step 1: Replace `sendMessage` with streaming version**

Full replacement of `frontend/components/ChatUI.tsx`:

```tsx
"use client";

import { useRef, useState } from "react";
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
}

interface Props {
  selectedEpisode: Episode | null;
}

export default function ChatUI({ selectedEpisode }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const threadId = useRef(crypto.randomUUID());
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Promise<void>>(Promise.resolve());

  function getAudioCtx(): AudioContext {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }

  async function enqueueAudioChunk(base64: string): Promise<void> {
    audioQueueRef.current = audioQueueRef.current.then(async () => {
      try {
        const ctx = getAudioCtx();
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const buffer = await ctx.decodeAudioData(bytes.buffer);
        await new Promise<void>((resolve) => {
          const src = ctx.createBufferSource();
          src.buffer = buffer;
          src.connect(ctx.destination);
          src.onended = () => resolve();
          src.start();
        });
      } catch {
        // audio decode failure is non-fatal — text still shows
      }
    });
  }

  async function sendMessage(userText: string) {
    if (!userText.trim()) return;
    setInput("");
    setLoading(true);

    const displayText = userText;
    const messagePayload = selectedEpisode
      ? `[Focus on episode: "${selectedEpisode.filename}" (id: ${selectedEpisode.id})] ${userText}`
      : userText;

    setMessages((prev) => [...prev, { role: "user", text: displayText }]);

    // Add assistant placeholder
    const assistantIdx = messages.length + 1;
    setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

    try {
      const res = await fetch("/api/agent/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messagePayload, thread_id: threadId.current }),
      });
      if (!res.ok || !res.body) throw new Error(await res.text());

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n\n");
        buf = lines.pop() ?? "";

        for (const line of lines) {
          const dataLine = line.startsWith("data: ") ? line.slice(6) : line;
          if (!dataLine.trim()) continue;
          try {
            const event = JSON.parse(dataLine);
            if (event.type === "token") {
              fullText += event.text;
              setMessages((prev) => {
                const next = [...prev];
                if (next[assistantIdx]) {
                  next[assistantIdx] = { ...next[assistantIdx], text: fullText };
                }
                return next;
              });
            } else if (event.type === "audio") {
              enqueueAudioChunk(event.data);
            } else if (event.type === "done") {
              break;
            }
          } catch {
            // malformed SSE line — skip
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, text: `Error: ${err}` };
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
            <span className="text-indigo-500 text-[10px]">deselect in sidebar</span>
          </div>
        </div>
      )}

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
                {m.role === "assistant" && loading && i === messages.length - 1 && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle" />
                )}
              </p>
              {m.sources && <SourcePlayer sources={m.sources} />}
            </div>
          </div>
        ))}

        {loading && messages[messages.length - 1]?.role !== "assistant" && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-2xl px-4 py-3 text-gray-400 text-sm animate-pulse">
              Thinking…
            </div>
          </div>
        )}
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

- [ ] **Step 2: Restart frontend container to pick up changes**

On Windows with Docker, inotify doesn't trigger on NTFS — must restart:

```bash
docker compose restart frontend
```

- [ ] **Step 3: Manual verification**

Open http://localhost:3000 and send a message. Verify:
1. The assistant text box appears immediately and text streams in token-by-token (blinking cursor visible)
2. Audio starts playing (in chunks) before the full response is done
3. Multiple audio chunks play sequentially without gaps
4. The focus-on-episode chip still works as before
5. Voice recording (AudioRecorder) still works — transcript gets sent to streaming endpoint

- [ ] **Step 4: Commit**

```bash
git add frontend/components/ChatUI.tsx
git commit -m "feat(frontend): streaming SSE chat with AudioContext gapless playback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AKFBDJAomY3MtdCnku8kQ6"
```

---

## Final Verification

After all 4 tasks complete, run the full backend test suite one last time:

```bash
cd backend
python -m pytest tests/ -v
```

Expected: all existing tests + new tests pass (134+ total).

Then squash per CLAUDE.md convention before moving to Etap 10:

```bash
git reset --soft $(git merge-base HEAD main)
git commit -m "feat(etap-9): streaming voice pipeline — SSE tokens + sentence TTS + AudioContext playback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AKFBDJAomY3MtdCnku8kQ6"
git push origin main
```

---

## Self-Review

**Spec coverage check:**
- ✅ Sentence chunker (Task 1) — `SentenceChunker` with boundary regex + MIN_WORDS
- ✅ `stream_agent_response` generator (Task 2) — `astream_events`, token events, TTS, base64 audio events, done event
- ✅ `POST /agent/chat/stream` endpoint (Task 3) — `StreamingResponse`, SSE media type
- ✅ Frontend streaming reader (Task 4) — `ReadableStream` reader, SSE parser, token-by-token text, AudioContext queue
- ✅ Gapless playback — promise-chained AudioContext queue ensures sequential playback

**No placeholders found.**

**Type consistency:** `SentenceChunker.push()` returns `Generator[str, None, None]` — used as `for sentence in chunker.push(token)` in `streaming.py` ✅. `stream_agent_response` yields `str` (SSE lines) — consumed as `async for chunk in stream_agent_response(...)` in router ✅.
