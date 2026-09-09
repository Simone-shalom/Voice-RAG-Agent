import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from unittest.mock import MagicMock, patch

import pytest

from app.ingest.summarizer import summarize_episode


def _make_tool_use_response(input_payload: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_payload
    message = MagicMock()
    message.content = [block]
    return message


def test_summarize_episode_raises_without_api_key():
    chunks = [{"start_ts": 0.0, "end_ts": 5.0, "text": "hello"}]
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            summarize_episode(chunks)


def test_summarize_episode_returns_structured_output():
    chunks = [
        {"start_ts": 0.0, "end_ts": 10.0, "text": "Welcome to the show, I'm Gary Jordan."},
        {"start_ts": 10.0, "end_ts": 60.0, "text": "Today we discuss NASA's new rocket."},
    ]
    payload = {
        "summary": "Gary Jordan hosts a discussion about NASA's new rocket program.",
        "chapters": [{"start_ts": 0.0, "title": "Introduction"}, {"start_ts": 10.0, "title": "NASA's new rocket"}],
        "speaker_names": {"Speaker 0": "Gary Jordan"},
    }
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = summarize_episode(chunks)

    assert result == payload
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "emit_episode_summary"}
    assert call_kwargs["tools"][0]["name"] == "emit_episode_summary"


def test_summarize_episode_empty_chunks_returns_empty_structure_without_api_call():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = summarize_episode([])
    assert result == {"summary": "", "chapters": [], "speaker_names": {}}


def test_summarize_episode_truncates_long_transcripts():
    long_text = "word " * 15000  # far exceeds MAX_TRANSCRIPT_CHARS (75,000 chars)
    chunks = [{"start_ts": 0.0, "end_ts": 10.0, "text": long_text}]
    payload = {"summary": "s", "chapters": [], "speaker_names": {}}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        summarize_episode(chunks)

    sent_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[...transcript truncated...]" in sent_content


def test_summarize_episode_includes_speaker_label_in_transcript_when_present():
    """
    speaker_names is only grounded if Claude can actually see which chunk
    belongs to which diarizer speaker index — the caller (ingest_audio) is
    expected to merge speaker_label onto each chunk dict before calling
    summarize_episode. Confirm that label actually reaches the prompt sent
    to Claude, prefixed on its line, so the model isn't guessing an index
    it was never shown.
    """
    chunks = [
        {"start_ts": 0.0, "end_ts": 10.0, "text": "Welcome to the show.", "speaker_label": "Speaker 0"},
        {"start_ts": 10.0, "end_ts": 20.0, "text": "Thanks for having me.", "speaker_label": "Speaker 1"},
    ]
    payload = {"summary": "s", "chapters": [], "speaker_names": {}}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        summarize_episode(chunks)

    sent_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[0.0s] Speaker 0: Welcome to the show." in sent_content
    assert "[10.0s] Speaker 1: Thanks for having me." in sent_content


def test_summarize_episode_omits_speaker_prefix_when_label_absent():
    """Chunks with no speaker_label key (or None) render exactly as before — no prefix."""
    chunks = [
        {"start_ts": 0.0, "end_ts": 10.0, "text": "No diarization ran."},
        {"start_ts": 10.0, "end_ts": 20.0, "text": "Still none.", "speaker_label": None},
    ]
    payload = {"summary": "s", "chapters": [], "speaker_names": {}}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        summarize_episode(chunks)

    sent_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[0.0s] No diarization ran." in sent_content
    assert "[10.0s] Still none." in sent_content
    assert "Speaker" not in sent_content


def test_summarize_episode_raises_if_no_tool_use_block_returned():
    chunks = [{"start_ts": 0.0, "end_ts": 5.0, "text": "hello"}]
    text_block = MagicMock()
    text_block.type = "text"
    message = MagicMock()
    message.content = [text_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = message

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with pytest.raises(RuntimeError, match="structured summary"):
            summarize_episode(chunks)
