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
