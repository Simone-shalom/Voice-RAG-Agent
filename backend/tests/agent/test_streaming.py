import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.streaming import stream_agent_response


def _make_token_event(text: str) -> dict:
    """
    Mocks the real Anthropic streaming shape (langchain-anthropic 1.x /
    langchain-core 1.x): `.content` is a list of content blocks, not a
    plain string. Verified against a live API call — see
    app.agent.streaming._extract_text's docstring for why this matters.
    """
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": MagicMock(content=[{"type": "text", "text": text, "index": 0}])},
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
async def test_ignores_tool_use_blocks_interleaved_with_text():
    """
    Regression test for the real bug: a tool-calling turn streams chunks
    like [{"type": "tool_use", ...}] and [{"type": "input_json_delta",
    "partial_json": "..."}] interleaved with [{"type": "text", "text":
    "..."}] blocks (and empty-list chunks at turn boundaries). Only the
    text blocks should become "token" events.
    """
    async def fake_stream(*args, **kwargs):
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content=[])}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(
            content=[{"id": "toolu_1", "input": {}, "name": "search_transcripts", "type": "tool_use"}]
        )}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(
            content=[{"partial_json": '{"query": "x"}', "type": "input_json_delta", "index": 1}]
        )}}
        yield _make_token_event("Real answer text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(
            stream_agent_response("hi", "t1", db=MagicMock())
        )

    token_events = [i for i in items if i["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["text"] == "Real answer text."


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
async def test_persists_user_message_and_accumulated_answer():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Hello ")
        yield _make_token_event("world.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch("app.agent.streaming.ensure_thread") as mock_ensure, \
         patch("app.agent.streaming.save_message") as mock_save, \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        db = MagicMock()
        await _collect(stream_agent_response("hi there", "t1", db=db))

    mock_ensure.assert_called_once_with(db, "t1", "hi there")
    assert mock_save.call_count == 2
    assert mock_save.call_args_list[0][0] == (db, "t1", "user", "hi there")
    assert mock_save.call_args_list[1][0] == (db, "t1", "assistant", "Hello world.", None)


@pytest.mark.asyncio
async def test_persists_assistant_message_with_sources():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Answer text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append({"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"})
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch("app.agent.streaming.ensure_thread"), \
         patch("app.agent.streaming.save_message") as mock_save, \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    assistant_call = mock_save.call_args_list[-1]
    assert assistant_call[0][2] == "assistant"
    assert assistant_call[0][3] == "Answer text."
    assert assistant_call[0][4] == [{"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"}]


@pytest.mark.asyncio
async def test_assistant_message_persisted_even_when_tts_fails():
    """
    Regression test: this shipped broken once — assistant-message
    persistence was originally placed *after* the TTS await loop, but a
    TTS failure is fatal-by-design (see test_tts_failure_yields_error_event_not_crash)
    and aborts the generator before reaching code after it. With
    ELEVENLABS_API_KEY unset (the real state during initial testing),
    every single streamed answer failed to persist. History must be saved
    before TTS is attempted, not after.
    """
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("The real answer.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", side_effect=RuntimeError("ElevenLabs API down")), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch("app.agent.streaming.ensure_thread"), \
         patch("app.agent.streaming.save_message") as mock_save, \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    assert any(i["type"] == "error" for i in items)  # TTS failure still surfaces
    assert mock_save.call_count == 2  # user + assistant both persisted anyway
    assert mock_save.call_args_list[1][0][2:4] == ("assistant", "The real answer.")


@pytest.mark.asyncio
async def test_history_persistence_failure_does_not_break_stream():
    """A history-write failure must never surface as a chat error event."""
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Real answer.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    with patch("app.agent.streaming.build_graph", return_value=mock_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch("app.agent.streaming.save_message", side_effect=RuntimeError("db down")), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    token_events = [i for i in items if i["type"] == "token"]
    assert "".join(t["text"] for t in token_events) == "Real answer."
    assert items[-1]["type"] == "done"
    assert not any(i["type"] == "error" for i in items)


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


@pytest.mark.asyncio
async def test_sources_event_strips_extra_fields_and_caps_at_max():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Short text.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append({
                "chunk_id": "c1", "episode_id": "ep1", "start_ts": 1.0,
                "end_ts": 2.0, "text": "hi", "similarity": 0.9,
            })
            for i in range(20):
                sources_sink.append({
                    "episode_id": f"ep{i}", "start_ts": float(i), "end_ts": float(i) + 1,
                    "text": f"chunk {i}",
                })
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):

        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    sources_event = next(i for i in items if i["type"] == "sources")
    assert sources_event["data"][0] == {"episode_id": "ep1", "start_ts": 1.0, "end_ts": 2.0, "text": "hi"}
    assert "chunk_id" not in sources_event["data"][0]
    assert "similarity" not in sources_event["data"][0]
    assert len(sources_event["data"]) == 8
