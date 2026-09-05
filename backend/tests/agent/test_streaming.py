import base64
import json
from unittest.mock import MagicMock, patch

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


@pytest.mark.asyncio
async def test_missing_api_key_yields_error():
    with patch.dict("os.environ", {}, clear=True):
        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )
    assert len(items) == 1
    assert items[0]["type"] == "error"


@pytest.mark.asyncio
async def test_tts_failure_yields_error_event_not_crash():
    long_sentence = (
        "The quick brown fox jumps over the lazy dog and then runs away. "
    )

    async def fake_stream(*args, **kwargs):
        yield _make_token_event(long_sentence)

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", side_effect=RuntimeError("ElevenLabs API down")), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )

    error_events = [i for i in items if i["type"] == "error"]
    assert len(error_events) == 1
    assert "ElevenLabs API down" in error_events[0]["text"]
    # must not have reached "done" after a failure
    assert not any(i["type"] == "done" for i in items)
