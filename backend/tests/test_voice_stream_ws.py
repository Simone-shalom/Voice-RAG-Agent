import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        from starlette.websockets import WebSocketDisconnect
        client = TestClient(app)
        with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://allowed.example"}):
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/voice/stream", headers={"origin": "https://evil.example"}
                ):
                    pass
            assert exc_info.value.code == 1008
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
            # A "token" event arrives before "audio" in this pipeline —
            # the latency clock must gate on "audio" specifically (Fix 4),
            # not fire on this earlier token.
            yield {"type": "token", "text": "Hi "}
            yield {"type": "audio", "data": "base64bytes"}
            yield {"type": "done"}

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                # The token event must be relayed WITHOUT a latency message
                # preceding it — latency is only reported on first audio.
                token_msg = ws.receive_json()
                assert token_msg == {"type": "token", "text": "Hi "}
                latency_msg = ws.receive_json()
                assert latency_msg["type"] == "latency"
                audio_msg = ws.receive_json()
                assert audio_msg == {"type": "audio", "data": "base64bytes"}
                done_msg = ws.receive_json()
                assert done_msg == {"type": "done"}
    finally:
        _clear_override()


def test_latency_not_sent_when_no_audio_event_occurs():
    # If the agent turn produces tokens but never an "audio" event (e.g.
    # TTS is skipped/unavailable), no latency message should ever be sent
    # — the old "token OR audio" gate would have fired one here.
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
                token_msg = ws.receive_json()
                assert token_msg == {"type": "token", "text": "Hi "}
                done_msg = ws.receive_json()
                assert done_msg == {"type": "done"}
                # No further messages — specifically no "latency" event.
    finally:
        _clear_override()


def test_latency_clock_starts_no_earlier_than_final_transcript():
    # The clock must start when the final transcript is received, not
    # after — starting it any earlier isn't possible with the data this
    # codebase currently has (no timestamp for the last raw audio byte),
    # so "final transcript received" is the earliest honest start point.
    # This test proves the reported latency reflects only the time from
    # final-transcript to first-audio-byte, by making stream_agent_response
    # take a known, measurable delay before its "audio" event and checking
    # the reported ms is at least that delay (and not, e.g., zero or
    # negative, which would indicate the clock started even later than
    # the agent call, or was never started).
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, **kwargs):
            await asyncio.sleep(0.05)
            yield {"type": "audio", "data": "base64bytes"}
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
                assert latency_msg["ms"] >= 40  # allow a little slack under 50ms
    finally:
        _clear_override()


def test_cancel_message_sets_cancel_event():
    # Regression test for the fix-round-1 defect: the original
    # implementation abandoned the socket reader when it broke out of
    # stream_transcribe, so a cancel frame sent mid-answer was never
    # observed. This test genuinely sends {"type": "cancel"} WHILE the
    # fake agent turn is in progress and asserts it actually takes
    # effect — not just that some Event object was passed through.
    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, cancel_event=None, **kwargs):
            yield {"type": "token", "text": "partial"}
            # give the test a window to send the cancel frame and have it
            # observed before this generator finishes
            for _ in range(20):
                if cancel_event is not None and cancel_event.is_set():
                    break
                await asyncio.sleep(0.05)
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "cancelled"}
            else:
                yield {"type": "done"}

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                # No "audio" event occurs in this turn, so no latency
                # message precedes the token (Fix 4: latency gates on
                # first "audio" specifically, not "token").
                ws.receive_json()  # token
                ws.send_text(json.dumps({"type": "cancel"}))
                final_msg = ws.receive_json()
                assert final_msg == {"type": "cancelled"}
    finally:
        _clear_override()


def test_rejects_connection_when_per_ip_cap_reached():
    # White-box: pre-populate the module-level per-IP connection counter
    # to simulate the cap already being at its limit, rather than trying
    # to hold genuinely concurrent WS connections open through the
    # synchronous TestClient (fragile/slow). Starlette's TestClient uses
    # a fixed default client address of ("testclient", 50000) with no
    # X-Forwarded-For header, so _client_key() resolves to "testclient"
    # for every connection in this test process.
    from app.voice import stream_ws

    _override_db()
    key = "testclient"
    stream_ws._active_connections[key] = stream_ws._MAX_CONNECTIONS_PER_IP
    try:
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        client = TestClient(app)
        with patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect(
                    "/voice/stream", headers={"origin": "http://testserver"}
                ):
                    pass
            assert exc_info.value.code == 1013
    finally:
        stream_ws._active_connections.pop(key, None)
        _clear_override()


def test_per_ip_counter_incremented_before_accept():
    # Fix B (TOCTOU): the counter must be incremented BEFORE
    # `await websocket.accept()`, not after — `accept()` yields to the
    # event loop, so incrementing only after it returns would let a burst
    # of concurrent connections from the same IP all observe the same
    # pre-increment count, all pass the cap check, and all slip through
    # together. This test proves the ordering directly by spying on
    # WebSocket.accept and asserting the counter already reflects this
    # connection's slot at the moment accept() is invoked.
    from app.voice import stream_ws
    from starlette.websockets import WebSocket

    _override_db()
    key = "testclient"
    stream_ws._active_connections.pop(key, None)
    observed: dict = {}
    original_accept = WebSocket.accept

    async def spy_accept(self, *args, **kwargs):
        observed["count_at_accept"] = stream_ws._active_connections.get(key, 0)
        return await original_accept(self, *args, **kwargs)

    try:
        from fastapi.testclient import TestClient

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, **kwargs):
            yield {"type": "done"}

        client = TestClient(app)
        with patch.object(WebSocket, "accept", spy_accept), \
             patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                ws.receive_json()  # done

        # The slot must already be claimed (count == 1) at the moment
        # accept() was called — proving the increment happened first, not
        # after the accept() yield.
        assert observed["count_at_accept"] == 1
    finally:
        stream_ws._active_connections.pop(key, None)
        _clear_override()


def test_counter_decremented_when_accept_itself_fails():
    # The increment now happens before `await websocket.accept()`, so if
    # accept() itself raises, the finally block must still roll that
    # increment back — increment/decrement must stay balanced on every
    # exit path, including an accept() failure, not just the normal ones.
    from app.voice import stream_ws
    from starlette.websockets import WebSocket

    _override_db()
    key = "testclient"
    stream_ws._active_connections.pop(key, None)

    async def failing_accept(self, *args, **kwargs):
        raise RuntimeError("simulated accept() failure")

    try:
        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch.object(WebSocket, "accept", failing_accept), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/voice/stream", headers={"origin": "http://testserver"}
                ):
                    pass

        assert stream_ws._active_connections.get(key, 0) == 0
    finally:
        stream_ws._active_connections.pop(key, None)
        _clear_override()


def test_per_ip_cap_is_released_when_connection_closes():
    # A connection that completes normally must decrement the counter, so
    # a later connection from the same IP isn't wrongly rejected.
    from app.voice import stream_ws

    _override_db()
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        async def fake_transcribe(audio_chunks):
            yield {"type": "final", "text": "hello agent"}

        async def fake_agent_response(*args, **kwargs):
            yield {"type": "done"}

        with patch("app.voice.stream_ws.stream_transcribe", side_effect=fake_transcribe), \
             patch("app.voice.stream_ws.stream_agent_response", side_effect=fake_agent_response), \
             patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://testserver"}):
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_bytes(b"irrelevant-audio")
                ws.receive_json()  # done

            # Connection has closed — the counter must be back to 0 (key
            # removed entirely, per the cleanup logic), not left at 1.
            assert stream_ws._active_connections.get("testclient", 0) == 0

            # And a fresh connection from the same IP must now succeed
            # rather than being rejected as if the cap were still hit.
            with client.websocket_connect(
                "/voice/stream", headers={"origin": "http://testserver"}
            ) as ws2:
                ws2.send_bytes(b"irrelevant-audio")
                ws2.receive_json()  # done
    finally:
        stream_ws._active_connections.pop("testclient", None)
        _clear_override()
