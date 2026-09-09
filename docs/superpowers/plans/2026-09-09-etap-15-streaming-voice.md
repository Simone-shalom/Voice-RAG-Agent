# Etap 15 — Streaming Voice: WebSocket Duplex, Barge-In, Measured Latency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current record-then-send voice flow with a real WebSocket duplex conversation — continuous mic streaming, real streaming STT (Deepgram), barge-in (interrupt the assistant mid-sentence), and a measured, displayed, logged latency number — with automatic fallback to the existing HTTP flow if the new path fails.

**Architecture:** New `WS /voice/stream` endpoint on the backend relays client audio to Deepgram's streaming STT, and on end-of-utterance drives the existing LangGraph agent + ElevenLabs TTS pipeline (`stream_agent_response`), pushing events back over the same socket. A new frontend `VoiceStream` component owns the mic capture, playback queue (abortable for barge-in), and a live latency badge, falling back to the existing `AudioRecorder` + HTTP flow if Deepgram or the socket is unavailable. `stream_agent_response` is refactored first to yield plain dicts instead of pre-formatted SSE strings, so both the existing SSE endpoint and the new WS endpoint can reuse the exact same generator without duplicating the LangGraph/TTS orchestration.

**Tech Stack:** FastAPI native WebSocket support (server side, no new dependency), `websockets` (new Python dependency, WS client to Deepgram), Deepgram streaming STT API (`DEEPGRAM_API_KEY`, new), existing LangGraph agent / ElevenLabs TTS / `SentenceChunker` (reused unchanged), MediaRecorder + native browser `WebSocket` (frontend, no new dependency).

**Spec:** `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md` (section "Etap 15 — Streaming voice: WebSocket duplex, barge-in, zmierzona latencja")

## Global Constraints

- No `Co-Authored-By: Claude Code` in commits; commit format `<type>(<scope>): <subject>`, scope `voice` or `frontend`.
- Backend tests are 100% mocked — never require a running database, a real Deepgram/Anthropic/ElevenLabs key, or a live WS server. Use FastAPI `TestClient.websocket_connect(...)` for WS route tests.
- Frontend tests use vitest + `@testing-library/react` + jsdom, per `CLAUDE.md`. jsdom has no real `MediaRecorder`, `WebSocket` audio decode, or `AudioContext` — stub what's needed in `frontend/vitest.setup.ts` (an `AudioContext` stub already exists there; extend it, don't replace it).
- Every raw-SQL-adjacent or streaming-shape assumption in this codebase has shipped broken once already (`:emb::vector` never binding, Anthropic's `content` being a list not a string) because mocked tests never exercised the real wire shape. Task 4 below explicitly calls out verifying Deepgram's real message shape live — do not skip that step even though it's not a red/green TDD step in the usual sense.
- `stream_agent_response`'s existing behavior (history persistence before TTS, TTS failure being fatal, sources deduplication/capping) must not regress — Task 1's refactor changes its *output format* only, never its logic.

---

### Task 1: Refactor `stream_agent_response` to yield dicts, not SSE strings

**Files:**
- Modify: `backend/app/agent/streaming.py`
- Modify: `backend/app/agent/router.py`
- Modify: `backend/tests/agent/test_streaming.py`

**Interfaces:**
- Produces: `stream_agent_response(...)` is now `AsyncGenerator[dict, None]` (was `AsyncGenerator[str, None]`). Event dict shapes are unchanged (`{"type": "token", "text": ...}` etc.) — only the SSE `data: ...\n\n` string wrapping moves out of this function.
- Produces: `_sse(payload: dict) -> str` stays in `streaming.py`, now used only by the router.

This is a pure refactor — no new behavior, no new test cases beyond updating the existing suite's collection helper. It exists so Task 5's WS endpoint can reuse this generator directly (`websocket.send_json(event)`) instead of re-parsing SSE strings.

- [ ] **Step 1: Update the failing test helper first**

In `backend/tests/agent/test_streaming.py`, replace `_collect`:

```python
async def _collect(gen) -> list[dict]:
    return [item async for item in gen]
```

Remove the now-unused `import json` at the top of the file if nothing else in it uses `json`.

- [ ] **Step 2: Run the suite to confirm it now fails**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v`
Expected: every test FAILS or errors, because `stream_agent_response` still yields `"data: {...}\n\n"` strings and `_collect` no longer unwraps them — items in the list are raw strings, not dicts, so `i["type"]` raises `TypeError`.

- [ ] **Step 3: Change `stream_agent_response` to yield dicts**

In `backend/app/agent/streaming.py`, replace every `yield _sse({...})` inside `stream_agent_response` with a bare `yield {...}` (same dict literal, no wrapper). There are five call sites: the error-on-missing-key early return, the per-token yield inside the `astream_events` loop, the sources event, the done event, and the error event in the `except` block. The `_sse` function itself and its docstring stay in the file unchanged — the router will use it directly.

```python
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "text": "ANTHROPIC_API_KEY not set"}
        return
```

```python
            answer_parts.append(token)
            yield {"type": "token", "text": token}
```

```python
        if sources_sink:
            yield {"type": "sources", "data": deduped}

        yield {"type": "done"}

    except Exception as exc:
        yield {"type": "error", "text": str(exc)}
```

Note the `audio` event is not a bare `yield` today — it's `yield await task` where `task` resolves to an SSE string from `_tts_chunk`. Change `_tts_chunk` to return the dict directly instead of calling `_sse`:

```python
    async def _tts_chunk(text: str) -> dict:
        loop = asyncio.get_running_loop()
        mp3_bytes = await loop.run_in_executor(None, synthesise, text)
        b64 = base64.b64encode(mp3_bytes).decode()
        return {"type": "audio", "data": b64, "text": text}
```

Update the type hint on `stream_agent_response` from `AsyncGenerator[str, None]` to `AsyncGenerator[dict, None]`, and update its docstring's opening line ("Async generator yielding SSE strings." → "Async generator yielding event dicts — the caller (SSE router or WebSocket handler) decides how to serialize them.").

- [ ] **Step 4: Update the router to wrap dicts back into SSE**

In `backend/app/agent/router.py`, add `from .streaming import stream_agent_response, _sse` (replacing the existing `from .streaming import stream_agent_response`), and change `chat_stream`'s inner generator:

```python
    async def generate():
        async for event in stream_agent_response(
            message=body.message,
            thread_id=body.thread_id,
            db=db,
            mode=body.mode,
        ):
            yield _sse(event)
```

- [ ] **Step 5: Run the full suite to confirm it passes**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass (245+), including every test in `tests/agent/test_streaming.py` and `tests/test_agent_router.py` unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/streaming.py backend/app/agent/router.py backend/tests/agent/test_streaming.py
git commit -m "refactor(agent): stream_agent_response yields dicts, not SSE strings

Pure output-format refactor — the SSE data:...\n\n wrapping moves to
the router (_sse is applied there now). No behavior change. This lets
the upcoming WS voice endpoint (Etap 15) reuse the exact same
generator via websocket.send_json() instead of re-parsing SSE text."
```

---

### Task 2: Barge-in cancellation support in `stream_agent_response`

**Files:**
- Modify: `backend/app/agent/streaming.py`
- Modify: `backend/tests/agent/test_streaming.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `stream_agent_response(message, thread_id, db, mode="hybrid", cancel_event: asyncio.Event | None = None)`. When `cancel_event` is set (checked once per streamed token), the generator stops consuming the graph, cancels any in-flight TTS tasks, yields `{"type": "cancelled"}`, and returns — it never reaches `"done"` and never runs the sources/persistence tail. `cancel_event=None` (the default, used by the existing SSE endpoint) preserves today's behavior exactly.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/agent/test_streaming.py`:

```python
@pytest.mark.asyncio
async def test_cancel_event_stops_stream_and_yields_cancelled():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("First. ")
        yield _make_token_event("Second sentence that keeps going. ")
        yield _make_token_event("Third sentence, should never be reached. ")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream
    cancel_event = asyncio.Event()

    async def cancel_after_first_token(gen):
        results = []
        async for item in gen:
            results.append(item)
            if item.get("type") == "token" and "First" in item["text"]:
                cancel_event.set()
        return results

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await cancel_after_first_token(
            stream_agent_response("hi", "t1", db=MagicMock(), cancel_event=cancel_event)
        )

    assert items[-1]["type"] == "cancelled"
    assert not any(i["type"] == "done" for i in items)
    assert not any("Third sentence" in i.get("text", "") for i in items)


@pytest.mark.asyncio
async def test_no_cancel_event_behaves_exactly_as_before():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Only sentence here, no cancellation.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    assert items[-1]["type"] == "done"
    assert not any(i["type"] == "cancelled" for i in items)
```

Add `import asyncio` to the top of the test file if not already present.

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py::test_cancel_event_stops_stream_and_yields_cancelled -v`
Expected: FAIL with `TypeError: stream_agent_response() got an unexpected keyword argument 'cancel_event'`.

- [ ] **Step 3: Implement cancellation**

In `backend/app/agent/streaming.py`, add the parameter and a check inside the token loop:

```python
async def stream_agent_response(
    message: str,
    thread_id: str,
    db: Session,
    mode: str = "hybrid",
    cancel_event: asyncio.Event | None = None,
) -> AsyncGenerator[dict, None]:
```

Inside the `try` block, right after `answer_parts.append(token)` / `yield {"type": "token", "text": token}` and the TTS-scheduling loop for that token, add the check before continuing to the next iteration:

```python
            answer_parts.append(token)
            yield {"type": "token", "text": token}

            for sentence in chunker.push(token):
                tts_tasks.append(asyncio.create_task(_tts_chunk(sentence)))

            if cancel_event is not None and cancel_event.is_set():
                for task in tts_tasks:
                    task.cancel()
                yield {"type": "cancelled"}
                return
```

Place this check as the last thing inside the `async for event in graph.astream_events(...)` loop body, after the existing token-handling logic and before the loop continues to its next iteration.

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v`
Expected: all tests in this file pass, including both new ones.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/streaming.py backend/tests/agent/test_streaming.py
git commit -m "feat(agent): optional cancel_event for barge-in interruption

stream_agent_response now accepts an asyncio.Event; when set mid-stream
it cancels pending TTS tasks, yields 'cancelled', and returns without
reaching 'done' or the persistence tail. Default (None) preserves
existing behavior exactly — the SSE endpoint doesn't pass one."
```

---

### Task 3: Structured latency logging helper

**Files:**
- Create: `backend/app/voice/latency.py`
- Test: `backend/tests/test_voice_latency.py`

**Interfaces:**
- Produces: `log_latency(stage: str, ms: float, thread_id: str) -> None` — logs one structured line via the module's own logger. `stage` is a short label like `"first_audio_byte"`.

- [ ] **Step 1: Write the failing test**

```python
import logging

from app.voice.latency import log_latency


def test_log_latency_emits_structured_log_line(caplog):
    with caplog.at_level(logging.INFO, logger="app.voice.latency"):
        log_latency("first_audio_byte", 642.5, "thread-abc")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.stage == "first_audio_byte"
    assert record.latency_ms == 642.5
    assert record.thread_id == "thread-abc"
    assert "642" in record.message
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_voice_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.voice.latency'`.

- [ ] **Step 3: Implement**

```python
import logging

logger = logging.getLogger(__name__)


def log_latency(stage: str, ms: float, thread_id: str) -> None:
    """
    Structured latency log line — stage is a short label (e.g.
    "first_audio_byte"), ms is the measured duration. Read these back
    with `docker compose logs backend | grep latency_ms` to compute
    p50/p95 for DECISIONS.md.
    """
    logger.info(
        "voice latency: %s = %.1fms (thread %s)",
        stage,
        ms,
        thread_id,
        extra={"stage": stage, "latency_ms": ms, "thread_id": thread_id},
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_voice_latency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/voice/latency.py backend/tests/test_voice_latency.py
git commit -m "feat(voice): structured latency logging helper

log_latency() emits one structured log line per measured stage so
real p50/p95 numbers can be pulled from logs after live testing,
instead of being a target nobody ever measured (Etap 4's original DoD)."
```

---

### Task 4: Deepgram streaming STT client

**Files:**
- Create: `backend/app/voice/deepgram_client.py`
- Test: `backend/tests/test_deepgram_client.py`
- Modify: `backend/requirements.txt` (add `websockets==13.1`)

**Interfaces:**
- Produces: `async def stream_transcribe(audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]` — connects to Deepgram's streaming endpoint, sends each item from `audio_chunks` as a binary WS frame, and yields parsed event dicts: `{"type": "interim", "text": str}` for partial results, `{"type": "final", "text": str}` when Deepgram marks an utterance's speech as final. Raises `RuntimeError` if `DEEPGRAM_API_KEY` is unset.

**IMPORTANT — verify before trusting this:** the exact JSON shape below (`channel.alternatives[0].transcript`, `is_final`, `speech_final`) is Deepgram's documented streaming response shape as of this plan's writing, but this project has shipped two bugs already (SQLAlchemy bind-param parsing, Anthropic's `content` list-vs-string) purely because a streaming/wire shape from documentation didn't match what mocked tests assumed. Before considering this task done, run a real 10-second manual test against the live Deepgram API (with a real `DEEPGRAM_API_KEY`) and print the raw JSON messages — if the shape differs, fix `_parse_message` and this docstring, and add a regression test using the *real* captured shape, the same way `test_ignores_tool_use_blocks_interleaved_with_text` was added after the Anthropic bug.

- [ ] **Step 1: Write the failing tests**

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.deepgram_client import stream_transcribe


async def _one_chunk_stream():
    yield b"fake-audio-bytes"


@pytest.mark.asyncio
async def test_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            async for _ in stream_transcribe(_one_chunk_stream()):
                pass


@pytest.mark.asyncio
async def test_yields_interim_and_final_events():
    interim_msg = json.dumps({
        "channel": {"alternatives": [{"transcript": "hel"}]},
        "is_final": False,
        "speech_final": False,
    })
    final_msg = json.dumps({
        "channel": {"alternatives": [{"transcript": "hello there"}]},
        "is_final": True,
        "speech_final": True,
    })

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.__aiter__.return_value = iter([interim_msg, final_msg])
    mock_ws.__aenter__.return_value = mock_ws
    mock_ws.__aexit__.return_value = False

    with patch("app.voice.deepgram_client.websockets.connect", return_value=mock_ws), \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):

        events = [e async for e in stream_transcribe(_one_chunk_stream())]

    assert {"type": "interim", "text": "hel"} in events
    assert {"type": "final", "text": "hello there"} in events


@pytest.mark.asyncio
async def test_ignores_empty_transcript_messages():
    empty_msg = json.dumps({
        "channel": {"alternatives": [{"transcript": ""}]},
        "is_final": False,
        "speech_final": False,
    })

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.__aiter__.return_value = iter([empty_msg])
    mock_ws.__aenter__.return_value = mock_ws
    mock_ws.__aexit__.return_value = False

    with patch("app.voice.deepgram_client.websockets.connect", return_value=mock_ws), \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):

        events = [e async for e in stream_transcribe(_one_chunk_stream())]

    assert events == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_deepgram_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.voice.deepgram_client'`.

- [ ] **Step 3: Add the dependency and implement**

In `backend/requirements.txt`, add a new line: `websockets==13.1`

```python
import json
import os
from typing import AsyncIterator

import websockets

DEEPGRAM_STREAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&punctuate=true&interim_results=true&endpointing=500"
)


def _parse_message(raw: str) -> dict | None:
    data = json.loads(raw)
    alternatives = data.get("channel", {}).get("alternatives", [])
    if not alternatives:
        return None
    text = alternatives[0].get("transcript", "")
    if not text:
        return None
    event_type = "final" if data.get("speech_final") else "interim"
    return {"type": event_type, "text": text}


async def stream_transcribe(audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    """
    Streams audio_chunks to Deepgram's streaming STT and yields
    {"type": "interim"|"final", "text": str} for each non-empty
    transcript Deepgram reports. Raises RuntimeError if
    DEEPGRAM_API_KEY is unset.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set — cannot use streaming voice")

    async with websockets.connect(
        DEEPGRAM_STREAM_URL,
        additional_headers={"Authorization": f"Token {api_key}"},
    ) as ws:
        async def _sender():
            async for chunk in audio_chunks:
                await ws.send(chunk)

        import asyncio
        sender_task = asyncio.create_task(_sender())
        try:
            async for raw in ws:
                event = _parse_message(raw)
                if event is not None:
                    yield event
        finally:
            sender_task.cancel()
```

Move `import asyncio` to the top of the file with the other imports rather than inline.

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && pip install -r requirements.txt && python -m pytest tests/test_deepgram_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/voice/deepgram_client.py backend/tests/test_deepgram_client.py backend/requirements.txt
git commit -m "feat(voice): Deepgram streaming STT client

Thin async wrapper over Deepgram's streaming WS API — sends audio
chunks, yields parsed interim/final transcript events. New
DEEPGRAM_API_KEY / websockets dependency. Message shape needs live
verification before Task 5 ships (see docstring) — this project has
shipped broken streaming-shape assumptions from documentation twice
already."
```

---

### Task 5: Shared allowed-origins helper (dedupe CORS parsing)

**Files:**
- Create: `backend/app/core/allowed_origins.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_allowed_origins.py`

**Interfaces:**
- Produces: `get_allowed_origins() -> list[str]` — parses `ALLOWED_ORIGINS` exactly as `main.py` does today.

This exists so the WS endpoint in Task 6 can validate the `Origin` header against the same list the CORS middleware already enforces for HTTP, without duplicating the parsing logic inline in two files.

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch

from app.core.allowed_origins import get_allowed_origins


def test_parses_comma_separated_origins():
    with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://a.com, https://b.com"}):
        assert get_allowed_origins() == ["https://a.com", "https://b.com"]


def test_defaults_to_localhost_3000():
    with patch.dict("os.environ", {}, clear=True):
        assert get_allowed_origins() == ["http://localhost:3000"]


def test_strips_blank_entries():
    with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://a.com,,  "}):
        assert get_allowed_origins() == ["https://a.com"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_allowed_origins.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement, then wire into `main.py`**

```python
import os


def get_allowed_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
```

In `backend/app/main.py`, replace the inline list comprehension with the shared helper:

```python
from .core.allowed_origins import get_allowed_origins
```

```python
_allowed_origins = get_allowed_origins()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_allowed_origins.py tests/ -v`
Expected: all pass, including the full suite (confirms `main.py`'s CORS behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/allowed_origins.py backend/app/main.py backend/tests/test_allowed_origins.py
git commit -m "refactor(core): extract ALLOWED_ORIGINS parsing into a shared helper

Was inlined in main.py only; the upcoming voice WS endpoint needs the
same list to validate the Origin header at handshake time, so it moved
to app/core/allowed_origins.py instead of duplicating the parsing."
```

---

### Task 6: `WS /voice/stream` endpoint

**Files:**
- Create: `backend/app/voice/stream_ws.py`
- Modify: `backend/app/voice/router.py`
- Test: `backend/tests/test_voice_stream_ws.py`

**Interfaces:**
- Consumes: `stream_transcribe` (Task 4), `stream_agent_response` with `cancel_event` (Task 2), `log_latency` (Task 3), `get_allowed_origins` (Task 5).
- Produces: `WS /voice/stream` — client sends binary audio frames continuously, and one JSON text frame `{"type": "cancel"}` to trigger barge-in. Server sends JSON text frames: every event `stream_agent_response` yields, plus `{"type": "latency", "ms": float}` right before the first `token`/`audio` event of each turn, plus `{"type": "fallback", "reason": str}` if Deepgram is unavailable (connection then closes and the client is expected to fall back to `/voice/stt`).

- [ ] **Step 1: Write the failing tests**

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app
from app.db.session import get_db


def _override_db():
    app.dependency_overrides[get_db] = lambda: MagicMock()


def _clear_override():
    app.dependency_overrides.clear()


def test_rejects_disallowed_origin():
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://allowed.example"}):
            try:
                with client.websocket_connect(
                    "/voice/stream", headers={"origin": "https://evil.example"}
                ):
                    pass
                assert False, "expected connection to be rejected"
            except Exception:
                pass  # starlette raises on the policy-violation close — rejection confirmed
    finally:
        _clear_override()


def test_deepgram_unavailable_sends_fallback_event():
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        async def failing_transcribe(audio_chunks):
            raise RuntimeError("DEEPGRAM_API_KEY not set — cannot use streaming voice")
            yield  # pragma: no cover — makes this an async generator

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=failing_transcribe), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                msg = ws.receive_json()
                assert msg["type"] == "fallback"
    finally:
        _clear_override()


def test_happy_path_relays_agent_events():
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, **kwargs):
            yield {"type": "token", "text": "Hi "}
            yield {"type": "done"}

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                latency_msg = ws.receive_json()
                assert latency_msg["type"] == "latency"
                token_msg = ws.receive_json()
                assert token_msg == {"type": "token", "text": "Hi "}
                done_msg = ws.receive_json()
                assert done_msg == {"type": "done"}
    finally:
        _clear_override()


def test_cancel_message_sets_cancel_event():
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        received_cancel_event = {}

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, cancel_event=None, **kwargs):
            received_cancel_event["event"] = cancel_event
            yield {"type": "token", "text": "partial"}
            yield {"type": "cancelled"}

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                ws.receive_json()  # latency
                ws.receive_json()  # token
                cancelled_msg = ws.receive_json()
                assert cancelled_msg == {"type": "cancelled"}
                assert received_cancel_event["event"] is not None
    finally:
        _clear_override()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_voice_stream_ws.py -v`
Expected: FAIL — `app.voice.stream_ws` doesn't exist yet, and `/voice/stream` isn't a registered route (404/connection-refused style failures).

- [ ] **Step 3: Implement**

```python
import asyncio
import json
import time

from fastapi import Depends, WebSocket, WebSocketDisconnect
from langchain_anthropic import ChatAnthropic
from sqlalchemy.orm import Session

from ..core.allowed_origins import get_allowed_origins
from ..db.session import get_db
from ..agent.streaming import stream_agent_response
from .deepgram_client import stream_transcribe
from .latency import log_latency


async def _client_audio(websocket: WebSocket, cancel_event: asyncio.Event):
    """
    Yields binary audio frames from the client. A JSON text frame
    {"type": "cancel"} sets cancel_event instead of being forwarded as
    audio — it's the barge-in signal, not speech.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if "bytes" in message and message["bytes"] is not None:
            yield message["bytes"]
        elif "text" in message and message["text"] is not None:
            try:
                payload = json.loads(message["text"])
            except ValueError:
                continue
            if payload.get("type") == "cancel":
                cancel_event.set()


async def websocket_voice_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    origin = websocket.headers.get("origin", "")
    if origin not in get_allowed_origins():
        await websocket.close(code=1008)  # policy violation
        return

    await websocket.accept()
    thread_id = websocket.query_params.get("thread_id", "voice-stream")
    cancel_event = asyncio.Event()

    try:
        transcript = ""
        async for event in stream_transcribe(_client_audio(websocket, cancel_event)):
            if event["type"] == "final":
                transcript = event["text"]
                break
    except RuntimeError as exc:
        await websocket.send_json({"type": "fallback", "reason": str(exc)})
        await websocket.close()
        return
    except WebSocketDisconnect:
        return

    if not transcript:
        await websocket.close()
        return

    start = time.monotonic()
    first_response_event_sent = False
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

    async for event in stream_agent_response(
        message=transcript, thread_id=thread_id, db=db, cancel_event=cancel_event
    ):
        if not first_response_event_sent and event["type"] in ("token", "audio"):
            elapsed_ms = (time.monotonic() - start) * 1000
            log_latency("first_response_event", elapsed_ms, thread_id)
            await websocket.send_json({"type": "latency", "ms": elapsed_ms})
            first_response_event_sent = True
        await websocket.send_json(event)

    await websocket.close()
```

Note: `stream_agent_response` already constructs its own `ChatAnthropic` internally in the current code — check `backend/app/agent/streaming.py` before writing this: if it does, do NOT pass `llm` here (remove the `llm = ChatAnthropic(...)` line above; it's dead code) — `stream_agent_response`'s existing signature is `(message, thread_id, db, mode="hybrid", cancel_event=None)` with no `llm` parameter, confirmed in Task 1/2. Delete that line before committing.

In `backend/app/voice/router.py`, mount the new route:

```python
from .stream_ws import websocket_voice_stream
```

```python
router.add_api_websocket_route("/stream", websocket_voice_stream)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_voice_stream_ws.py tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/voice/stream_ws.py backend/app/voice/router.py backend/tests/test_voice_stream_ws.py
git commit -m "feat(voice): WS /voice/stream endpoint — Deepgram + agent + barge-in

Validates Origin against ALLOWED_ORIGINS at handshake. Relays client
audio to Deepgram streaming STT; on a final transcript, drives the
existing agent/TTS pipeline over the same socket. A 'cancel' text
frame from the client sets the cancel_event that Task 2 wired into
stream_agent_response — that's the server side of barge-in. Falls back
(sends {type: fallback} and closes) if Deepgram is unavailable."
```

---

### Task 7: Frontend WebSocket connection helper

**Files:**
- Create: `frontend/lib/voiceWebSocket.ts`
- Test: `frontend/lib/__tests__/voiceWebSocket.test.ts`

**Interfaces:**
- Produces: `voiceStreamUrl(threadId: string): string` — converts `NEXT_PUBLIC_API_URL` (already client-visible per its `NEXT_PUBLIC_` prefix, already read in `next.config.ts`) from `http(s)://` to `ws(s)://` and appends `/voice/stream?thread_id=...`. Connects directly to the backend, **not** through the `/api/*` Next.js rewrite proxy in `next.config.ts` — that proxy is HTTP-only and does not reliably forward a WebSocket upgrade handshake.
- Produces: `connectVoiceStream(threadId: string, handlers: { onMessage: (event: object) => void; onOpen?: () => void; onClose?: () => void }): { send: (data: Blob | string) => void; close: () => void }` — thin wrapper with reconnect-with-backoff (max 3 attempts, 500ms/1500ms/4000ms delays) on unexpected close.

- [ ] **Step 1: Write the failing tests**

```typescript
import { describe, expect, it, vi, beforeEach } from "vitest";
import { voiceStreamUrl, connectVoiceStream } from "../voiceWebSocket";

describe("voiceStreamUrl", () => {
  it("converts https to wss and appends the path", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://backend.example.com");
    expect(voiceStreamUrl("t1")).toBe("wss://backend.example.com/voice/stream?thread_id=t1");
  });

  it("converts http to ws", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:8000");
    expect(voiceStreamUrl("t1")).toBe("ws://localhost:8000/voice/stream?thread_id=t1");
  });
});

describe("connectVoiceStream", () => {
  let sockets: any[];

  beforeEach(() => {
    sockets = [];
    vi.stubGlobal(
      "WebSocket",
      vi.fn().mockImplementation(function (this: any) {
        this.send = vi.fn();
        this.close = vi.fn();
        sockets.push(this);
        return this;
      })
    );
  });

  it("forwards parsed JSON messages to onMessage", () => {
    const onMessage = vi.fn();
    connectVoiceStream("t1", { onMessage });
    const socket = sockets[0];
    socket.onmessage({ data: JSON.stringify({ type: "token", text: "hi" }) });
    expect(onMessage).toHaveBeenCalledWith({ type: "token", text: "hi" });
  });

  it("send() forwards to the underlying socket", () => {
    const controls = connectVoiceStream("t1", { onMessage: vi.fn() });
    controls.send("hello");
    expect(sockets[0].send).toHaveBeenCalledWith("hello");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npm test -- voiceWebSocket`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```typescript
const RECONNECT_DELAYS_MS = [500, 1500, 4000];

export function voiceStreamUrl(threadId: string): string {
  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/voice/stream?thread_id=${encodeURIComponent(threadId)}`;
}

interface Handlers {
  onMessage: (event: Record<string, unknown>) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export function connectVoiceStream(threadId: string, handlers: Handlers) {
  let socket: WebSocket | null = null;
  let attempt = 0;
  let deliberatelyClosed = false;

  function open() {
    socket = new WebSocket(voiceStreamUrl(threadId));
    socket.onopen = () => {
      attempt = 0;
      handlers.onOpen?.();
    };
    socket.onmessage = (evt: MessageEvent) => {
      try {
        handlers.onMessage(JSON.parse(evt.data));
      } catch {
        // non-JSON frame — ignore
      }
    };
    socket.onclose = () => {
      handlers.onClose?.();
      if (deliberatelyClosed || attempt >= RECONNECT_DELAYS_MS.length) return;
      const delay = RECONNECT_DELAYS_MS[attempt];
      attempt += 1;
      setTimeout(open, delay);
    };
  }

  open();

  return {
    send(data: Blob | string) {
      socket?.send(data as never);
    },
    close() {
      deliberatelyClosed = true;
      socket?.close();
    },
  };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npm test -- voiceWebSocket`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/voiceWebSocket.ts frontend/lib/__tests__/voiceWebSocket.test.ts
git commit -m "feat(frontend): voiceWebSocket connection helper with reconnect backoff

Connects directly to the backend's ws(s):// origin (derived from the
already-client-visible NEXT_PUBLIC_API_URL) — the Next.js rewrite
proxy in next.config.ts is HTTP-only and won't forward a WS upgrade."
```

---

### Task 8: Extract `applyStreamEvent` from ChatUI (shared by SSE and WS paths)

**Files:**
- Modify: `frontend/components/ChatUI.tsx`
- Test: `frontend/components/__tests__/applyStreamEvent.test.ts` (new)

**Interfaces:**
- Produces: `applyStreamEvent(event: Record<string, unknown>, setMessages: React.Dispatch<React.SetStateAction<Message[]>>, onAudio: (base64: string) => void) => boolean` — applies one stream event (token/audio/sources/error/done/cancelled) to the messages state exactly as ChatUI's SSE reader loop does today. Returns `true` if this event ends the turn (done/error/cancelled), so callers know when to stop.

This task only extracts existing logic into a standalone, exported, independently-testable function — it changes nothing about what the app does. Task 9 wires `VoiceStream` to reuse it.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it, vi } from "vitest";
import { applyStreamEvent, type Message } from "../ChatUI";

function seedMessages(): Message[] {
  return [
    { role: "user", text: "hi" },
    { role: "assistant", text: "", streaming: true },
  ];
}

describe("applyStreamEvent", () => {
  it("appends token text to the last assistant message", () => {
    let messages = seedMessages();
    const setMessages = (updater: any) => { messages = updater(messages); };
    applyStreamEvent({ type: "token", text: "Hello" }, setMessages, vi.fn());
    expect(messages[1].text).toBe("Hello");
  });

  it("calls onAudio for audio events", () => {
    const messages = seedMessages();
    const onAudio = vi.fn();
    applyStreamEvent({ type: "audio", data: "base64data" }, () => {}, onAudio);
    expect(onAudio).toHaveBeenCalledWith("base64data");
  });

  it("returns true and marks streaming false on done", () => {
    let messages = seedMessages();
    const setMessages = (updater: any) => { messages = updater(messages); };
    const isTerminal = applyStreamEvent({ type: "done" }, setMessages, vi.fn());
    expect(isTerminal).toBe(true);
    expect(messages[1].streaming).toBe(false);
  });

  it("returns true on cancelled, leaving partial text intact", () => {
    let messages = seedMessages();
    messages[1].text = "partial answer";
    const setMessages = (updater: any) => { messages = updater(messages); };
    const isTerminal = applyStreamEvent({ type: "cancelled" }, setMessages, vi.fn());
    expect(isTerminal).toBe(true);
    expect(messages[1].text).toBe("partial answer");
    expect(messages[1].streaming).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npm test -- applyStreamEvent`
Expected: FAIL — `applyStreamEvent` and `Message` are not exported from `ChatUI.tsx` yet.

- [ ] **Step 3: Extract the function**

In `frontend/components/ChatUI.tsx`, export the existing `Message`/`Source` interfaces (add `export` in front of both), then add a new exported function above the `ChatUI` component, built from the exact logic already inline in `sendMessage`'s SSE-parsing loop:

```typescript
export function applyStreamEvent(
  event: { type: string; text?: string; data?: string | Source[] },
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  onAudio: (base64: string) => void
): boolean {
  if (event.type === "token") {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, text: last.text + ((event.text as string) ?? "") };
      }
      return next;
    });
    return false;
  }
  if (event.type === "audio" && typeof event.data === "string") {
    onAudio(event.data);
    return false;
  }
  if (event.type === "sources" && Array.isArray(event.data)) {
    const sources = event.data;
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, sources };
      }
      return next;
    });
    return false;
  }
  if (event.type === "error") {
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
    return true;
  }
  if (event.type === "done" || event.type === "cancelled") {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, streaming: false };
      }
      return next;
    });
    return true;
  }
  return false;
}
```

Then replace the body of the SSE-parsing `for (const raw of events)` loop inside `sendMessage` to call this function instead of repeating the logic:

```typescript
          let event: { type: string; text?: string; data?: string | Source[] };
          try {
            event = JSON.parse(line);
          } catch {
            continue;
          }

          if (applyStreamEvent(event, setMessages, enqueueAudioChunk)) {
            done = true;
          }
```

Delete the old inline `if (event.type === "token") {...} else if (...) {...}` chain that this replaces — everything from the `if (event.type === "token")` line through the final `else if (event.type === "done") { done = true; }` line.

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npm test`
Expected: all frontend tests pass (11 existing + 4 new), and the existing `ChatHistory.test.tsx` / `ChatUI.autoscroll.test.tsx` tests still pass unchanged — this confirms the extraction didn't change `sendMessage`'s observable behavior.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/ChatUI.tsx frontend/components/__tests__/applyStreamEvent.test.ts
git commit -m "refactor(frontend): extract applyStreamEvent from ChatUI's SSE loop

Pure extraction, no behavior change (existing ChatUI tests still pass
unchanged) — makes the event-handling logic reusable by the upcoming
VoiceStream WS path instead of duplicating the token/audio/sources/
error/done/cancelled handling a second time."
```

---

### Task 9: `VoiceStream` component — mic streaming, playback abort, latency badge, fallback

**Files:**
- Create: `frontend/components/VoiceStream.tsx`
- Test: `frontend/components/__tests__/VoiceStream.test.tsx`
- Modify: `frontend/vitest.setup.ts` (stub `MediaRecorder`)
- Modify: `frontend/components/ChatUI.tsx` (render `VoiceStream` instead of `AudioRecorder`)

**Interfaces:**
- Consumes: `connectVoiceStream` (Task 7), `applyStreamEvent` (Task 8).
- Produces: `<VoiceStream setMessages={...} threadId={...} />` — on mount, opens the WS connection and starts mic capture (`MediaRecorder` with a 250ms `timeslice`, sending each blob as a binary frame). On a `{"type": "latency", "ms": N}` message, shows a badge. On any assistant `token`/`audio` event arriving while a previous turn's audio is still enqueued, calls the same abort path a `cancel` message triggers, so playback stops immediately. On `{"type": "fallback"}` or a connection that never opens, renders the existing `AudioRecorder` instead.

- [ ] **Step 1: Add the `MediaRecorder` stub**

In `frontend/vitest.setup.ts`, add alongside the existing `AudioContext` stub (do not remove that one):

```typescript
class FakeMediaRecorder {
  static isTypeSupported() {
    return true;
  }
  state: "inactive" | "recording" = "inactive";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(_stream: MediaStream) {}
  start(_timeslice?: number) {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.onstop?.();
  }
}
// @ts-expect-error — partial stub, sufficient for tests that don't assert on actual encoded audio
globalThis.MediaRecorder = FakeMediaRecorder;

// @ts-expect-error — jsdom has no getUserMedia; tests stub it per-test via vi.stubGlobal on navigator.mediaDevices
globalThis.navigator.mediaDevices = globalThis.navigator.mediaDevices ?? {};
```

- [ ] **Step 2: Write the failing tests**

```typescript
import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, expect, it, beforeEach } from "vitest";
import VoiceStream from "../VoiceStream";

vi.mock("../../lib/voiceWebSocket", () => ({
  connectVoiceStream: vi.fn(),
}));

import { connectVoiceStream } from "../../lib/voiceWebSocket";

describe("VoiceStream", () => {
  let onMessageHandler: (event: Record<string, unknown>) => void;

  beforeEach(() => {
    (connectVoiceStream as any).mockImplementation((_threadId: string, handlers: any) => {
      onMessageHandler = handlers.onMessage;
      handlers.onOpen?.();
      return { send: vi.fn(), close: vi.fn() };
    });
    (globalThis.navigator.mediaDevices as any).getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
  });

  it("shows a latency badge on a latency event", async () => {
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);
    onMessageHandler({ type: "latency", ms: 642 });
    await waitFor(() => expect(screen.getByText(/642/)).toBeInTheDocument());
  });

  it("falls back to AudioRecorder UI on a fallback event", async () => {
    render(<VoiceStream setMessages={vi.fn()} threadId="t1" />);
    onMessageHandler({ type: "fallback", reason: "Deepgram unavailable" });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /ask with voice/i })).toBeInTheDocument()
    );
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd frontend && npm test -- VoiceStream`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement**

```typescript
"use client";

import { useEffect, useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import { connectVoiceStream } from "@/lib/voiceWebSocket";
import { applyStreamEvent, type Message } from "./ChatUI";

interface Props {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  threadId: string;
  onTranscript?: (text: string) => void;
}

export default function VoiceStream({ setMessages, threadId }: Props) {
  const [fallback, setFallback] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const controlsRef = useRef<{ send: (d: Blob | string) => void; close: () => void } | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const answeringRef = useRef(false);

  function abortPlayback() {
    currentSourceRef.current?.stop();
    currentSourceRef.current = null;
  }

  async function playAudioChunk(base64: string) {
    const ctx = audioCtxRef.current ?? new AudioContext();
    audioCtxRef.current = ctx;
    if (ctx.state === "suspended") await ctx.resume();
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    try {
      const buffer = await ctx.decodeAudioData(bytes.buffer);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      currentSourceRef.current = src;
      src.start();
    } catch {
      // decode/playback failure is non-fatal
    }
  }

  useEffect(() => {
    if (fallback) return;

    const controls = connectVoiceStream(threadId, {
      onMessage: (event) => {
        if (event.type === "fallback") {
          setFallback(true);
          return;
        }
        if (event.type === "latency" && typeof event.ms === "number") {
          setLatencyMs(event.ms);
        }
        if ((event.type === "token" || event.type === "audio") && !answeringRef.current) {
          // first event of a new turn — a prior turn's playback (if any)
          // must stop immediately, this is the barge-in path
          abortPlayback();
          answeringRef.current = true;
        }
        const isTerminal = applyStreamEvent(
          event as { type: string; text?: string; data?: string | never[] },
          setMessages,
          playAudioChunk
        );
        if (isTerminal) answeringRef.current = false;
      },
    });
    controlsRef.current = controls;

    let recorder: MediaRecorder | null = null;
    let stream: MediaStream | null = null;
    navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
      stream = s;
      recorder = new MediaRecorder(s);
      recorder.ondataavailable = (e) => controlsRef.current?.send(e.data);
      recorder.start(250);
    }).catch(() => setFallback(true));

    return () => {
      controls.close();
      recorder?.stop();
      stream?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close().catch(() => {});
    };
  }, [threadId, fallback, setMessages]);

  if (fallback) {
    return <AudioRecorder onTranscript={() => {}} />;
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <p className="text-xs text-gray-500">🎙️ Listening — talk any time, even mid-answer</p>
      {latencyMs !== null && (
        <p className="text-[10px] text-gray-600">⚡ {Math.round(latencyMs)}ms</p>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd frontend && npm test -- VoiceStream`
Expected: PASS.

- [ ] **Step 6: Wire into ChatUI**

In `frontend/components/ChatUI.tsx`, replace the `<AudioRecorder onTranscript={(t) => sendMessage(t)} />` line in the input bar with:

```tsx
<VoiceStream setMessages={setMessages} threadId={threadId} />
```

Add the import: `import VoiceStream from "./VoiceStream";`. Leave the `AudioRecorder` import in place — `VoiceStream` itself renders it as its fallback.

- [ ] **Step 7: Run full frontend suite**

Run: `cd frontend && npm test`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add frontend/components/VoiceStream.tsx frontend/components/__tests__/VoiceStream.test.tsx frontend/vitest.setup.ts frontend/components/ChatUI.tsx
git commit -m "feat(frontend): VoiceStream duplex voice UI — mic streaming, barge-in, latency badge

Replaces AudioRecorder as the default voice UI: continuous mic capture
over WS, immediate playback abort when a new turn's first event
arrives (barge-in), a live latency badge, and automatic fallback to
the existing AudioRecorder + HTTP flow on a {type: fallback} event or
a getUserMedia failure."
```

---

### Task 10: Live verification, env docs, DECISIONS.md

**Files:**
- Modify: `.env.example`, `CLAUDE.md` (env var table + gotchas), `DECISIONS.md`
- No test file — this task's deliverable is manual verification evidence + documentation.

- [ ] **Step 1: Add `DEEPGRAM_API_KEY` to `.env.example`**

Add a line following the existing style (Polish comment, matching the other entries):

```
# Streaming STT dla rozmowy głosowej duplex (Etap 15). Bez tego /voice/stream
# wysyła zdarzenie fallback i klient wraca do starego /voice/stt.
DEEPGRAM_API_KEY=
```

- [ ] **Step 2: Add the row to `CLAUDE.md`'s environment variable table**

Add a row: `| DEEPGRAM_API_KEY | Streaming voice duplex (Etap 15) — optional, /voice/stream falls back to /voice/stt without it |`

- [ ] **Step 3: Manual smoke test — barge-in**

Run `docker compose up`, open the app, start a voice conversation, and while the assistant is mid-answer, speak again. Confirm the playback audibly stops within roughly one Deepgram interim-result round-trip. Record the observed behavior (worked / didn't / what you saw) — this is the DoD's barge-in requirement and cannot be verified by an automated test in this codebase's mocked-only test policy.

- [ ] **Step 4: Manual smoke test — fallback**

Temporarily unset `DEEPGRAM_API_KEY` (or set it to an invalid value) in `.env`, restart the backend, and confirm the UI falls back to the `AudioRecorder` button/flow instead of silently failing. Restore the real key afterward.

- [ ] **Step 5: Collect real latency numbers**

With a valid `DEEPGRAM_API_KEY`, have 10-15 real voice exchanges through the app. Pull the logged values with `docker compose logs backend | grep "voice latency"`, compute p50 and p95 by hand or with a one-line script, and write them down.

- [ ] **Step 6: Write the `DECISIONS.md` entry**

Follow the existing entry format in the file (see any `## [Etap N] ...` entry for the exact structure: Decyzja / Alternatywy odrzucone / Uzasadnienie). Include the real p50/p95 numbers from Step 5, not a target — this is the entire point of this etap per its own spec section ("nie deklaracją celu").

- [ ] **Step 7: Commit**

```bash
git add .env.example CLAUDE.md DECISIONS.md
git commit -m "docs(etap-15): env var, gotcha, and real measured latency numbers

Closes the loop this etap exists for — DECISIONS.md now has actual
p50/p95 latency from live testing, not the never-measured <3s target
from Etap 4's original DoD."
```

---

## Self-Review Notes

- **Spec coverage:** WebSocket duplex (Task 6), streaming STT via Deepgram (Task 4), barge-in (Tasks 2, 6, 9), measured + displayed + logged latency (Tasks 3, 6, 9, 10), fallback to the existing HTTP path (Tasks 6, 9, 10) — all five bullet points from the roadmap's "Zakres" are covered. The DoD's four numbered items map to Task 10 (items 1, 2, 3 are manual verification steps; item 4 is the `DECISIONS.md` write-up).
- **Out-of-scope items honored:** no telephony, no language switching, no wake-word, Whisper is untouched for ingest — none of Tasks 1-10 touch `backend/app/ingest/`.
- **Type/name consistency checked:** `stream_agent_response`'s new `cancel_event` parameter (Task 2) is consumed identically in Task 6's endpoint; `applyStreamEvent`'s signature (Task 8) matches exactly how Task 9's `VoiceStream` calls it; `connectVoiceStream`'s `{ send, close }` return shape (Task 7) matches how Task 9 uses `controlsRef.current`.
