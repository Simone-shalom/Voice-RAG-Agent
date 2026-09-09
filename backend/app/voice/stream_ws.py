import asyncio
import contextlib
import json
import time

from fastapi import Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from ..core.allowed_origins import get_allowed_origins
from ..db.session import get_db
from ..agent.streaming import stream_agent_response
from .deepgram_client import stream_transcribe
from .latency import log_latency

# Minimal abuse guard for a WS endpoint that has no rate limiting
# (slowapi's decorators don't apply to WebSocket routes) and only
# browser-enforced Origin validation (trivially bypassed by a non-browser
# client that just sets an arbitrary Origin header). Every accepted
# connection opens a billed Deepgram stream and can drive further billed
# Anthropic/ElevenLabs calls, so this bounds the blast radius of casual
# abuse without adding real auth (out of scope here) — same spirit as
# `core.rate_limit`/`API_KEY` for the HTTP endpoints.
_MAX_CONNECTIONS_PER_IP = 2
_MAX_CONNECTION_SECONDS = 300  # 5 minutes — bounds worst-case cost of one stuck/abusive connection

# client_key -> count of currently-open connections from that key. Simple
# in-memory dict is enough here: no persistence, no user accounts, single
# backend process (matches the project's existing non-distributed deploy).
_active_connections: dict[str, int] = {}


def _client_key(websocket: WebSocket) -> str:
    """
    Same X-Forwarded-For-aware IP extraction as
    `app.core.rate_limit._client_key`, adapted for a WebSocket (no
    `Request` object to call slowapi's `get_remote_address` on) — prefers
    the original client IP from X-Forwarded-For (set by a reverse proxy,
    e.g. the Next.js rewrite in dev, or a production edge/load balancer)
    over the raw TCP peer, which would otherwise be the proxy itself for
    every connection and collapse all users into one shared bucket.
    """
    forwarded_for = websocket.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    client = websocket.client
    return client.host if client is not None else "unknown"


async def _read_client_forever(
    websocket: WebSocket, queue: "asyncio.Queue[bytes]", cancel_event: asyncio.Event
) -> None:
    """
    The ONE reader of this socket, for the whole connection lifetime — not
    just during transcription. Pushes binary audio frames onto `queue`;
    on a JSON text frame {"type": "cancel"} sets cancel_event instead of
    queueing it (that's the barge-in signal, not speech). Exits cleanly on
    a client disconnect.

    This must run as an independent task, NOT be handed directly to
    stream_transcribe as its audio source: stream_transcribe's internal
    sender task is cancelled (in its own `finally`) the moment its caller
    stops iterating it (e.g. on the first "final" transcript). If this
    receive-loop lived inside that generator's ownership, a cancel frame
    sent mid-answer — after transcription has already handed off to the
    agent — would sit unread on the socket forever and barge-in would be
    dead. Decoupling via `queue` keeps this loop alive through the entire
    agent-response turn.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if "bytes" in message and message["bytes"] is not None:
            await queue.put(message["bytes"])
        elif "text" in message and message["text"] is not None:
            try:
                payload = json.loads(message["text"])
            except ValueError:
                continue
            if payload.get("type") == "cancel":
                cancel_event.set()


async def _dequeue_audio(queue: "asyncio.Queue[bytes]"):
    """
    stream_transcribe's actual audio source: drains `queue` (fed by
    `_read_client_forever`). This is the generator stream_transcribe owns
    and abandons on break — the socket reader above is untouched by that.
    """
    while True:
        yield await queue.get()


async def _run_voice_turn(websocket: WebSocket, db: Session) -> None:
    """
    One full connection's worth of work: transcribe -> agent -> relay.
    Split out from `websocket_voice_stream` so the latter can wrap this in
    an `asyncio.wait_for(..., timeout=_MAX_CONNECTION_SECONDS)` without
    tangling the per-IP connection-count bookkeeping into the timeout
    logic.
    """
    thread_id = websocket.query_params.get("thread_id", "voice-stream")
    cancel_event = asyncio.Event()
    audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()

    reader_task = asyncio.create_task(
        _read_client_forever(websocket, audio_queue, cancel_event)
    )

    try:
        try:
            transcript = ""
            start: float | None = None
            async with contextlib.aclosing(
                stream_transcribe(_dequeue_audio(audio_queue))
            ) as transcribed:
                async for event in transcribed:
                    if event["type"] == "final":
                        transcript = event["text"]
                        # "End of user utterance" clock starts here — the
                        # earliest point this codebase can honestly call
                        # that, since it doesn't yet timestamp the actual
                        # last audio byte received from the client. This
                        # already includes Deepgram's endpointing delay
                        # (endpointing=500 in deepgram_client.py), unlike
                        # starting the clock only after the agent call
                        # begins.
                        start = time.monotonic()
                        break
        except RuntimeError as exc:
            await websocket.send_json({"type": "fallback", "reason": str(exc)})
            return

        if not transcript:
            return

        first_audio_sent = False

        async for event in stream_agent_response(
            message=transcript, thread_id=thread_id, db=db, cancel_event=cancel_event
        ):
            # Gate specifically on the first AUDIO byte, not the first
            # token — the spec is "time to spoken response", and tokens
            # arrive before audio in this pipeline, so gating on either
            # one (as before) systematically under-reports real latency.
            if not first_audio_sent and event["type"] == "audio" and start is not None:
                elapsed_ms = (time.monotonic() - start) * 1000
                log_latency("first_audio_byte", elapsed_ms, thread_id)
                await websocket.send_json({"type": "latency", "ms": elapsed_ms})
                first_audio_sent = True
            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        with contextlib.suppress(BaseException):
            await reader_task


async def websocket_voice_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    origin = websocket.headers.get("origin", "")
    if origin not in get_allowed_origins():
        await websocket.close(code=1008)  # policy violation
        return

    client_key = _client_key(websocket)
    if _active_connections.get(client_key, 0) >= _MAX_CONNECTIONS_PER_IP:
        await websocket.close(code=1013)  # try again later
        return

    # Increment BEFORE the `await websocket.accept()` below, not after.
    # `accept()` yields to the event loop, so if the increment happened
    # after it, a burst of N concurrent connections from the same IP could
    # all read the same pre-increment count, all pass the cap check above,
    # all yield at accept(), and only then all increment — the cap
    # wouldn't actually bound a concurrent burst (exactly the abuse
    # pattern it exists to stop). Incrementing first closes that TOCTOU
    # window: any connection that reaches here has already claimed its
    # slot, so a concurrent burst beyond the cap gets rejected above
    # instead of all slipping through together. Sequential/normal
    # reconnect traffic is unaffected since those requests aren't
    # concurrent with each other.
    _active_connections[client_key] = _active_connections.get(client_key, 0) + 1
    try:
        try:
            await websocket.accept()
            await asyncio.wait_for(
                _run_voice_turn(websocket, db), timeout=_MAX_CONNECTION_SECONDS
            )
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass
    finally:
        remaining = _active_connections.get(client_key, 0) - 1
        if remaining <= 0:
            _active_connections.pop(client_key, None)
        else:
            _active_connections[client_key] = remaining
        with contextlib.suppress(Exception):
            await websocket.close()
