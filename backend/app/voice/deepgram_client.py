import asyncio
import json
import os
from typing import AsyncIterator

import websockets

DEEPGRAM_STREAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&punctuate=true&interim_results=true&endpointing=500"
)


def _parse_message(raw: str) -> dict | None:
    # IMPORTANT — verify before trusting this: this shape
    # (channel.alternatives[0].transcript, is_final, speech_final) is
    # Deepgram's documented streaming response shape as of this plan's
    # writing, but this project has shipped two bugs already (SQLAlchemy
    # bind-param parsing, Anthropic's `content` list-vs-string) purely
    # because a streaming/wire shape from documentation didn't match what
    # mocked tests assumed. Before trusting this in production, run a real
    # manual test against the live Deepgram API with a real
    # DEEPGRAM_API_KEY and print the raw JSON messages — if the shape
    # differs, fix this function and this docstring, and add a regression
    # test using the *real* captured shape.
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

        sender_task = asyncio.create_task(_sender())
        try:
            async for raw in ws:
                event = _parse_message(raw)
                if event is not None:
                    yield event
        finally:
            sender_task.cancel()
