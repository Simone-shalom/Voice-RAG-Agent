import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
import pytest
from app.voice.stt import transcribe_audio


def make_mock_client(text="hello world"):
    mock = MagicMock()
    mock.audio.transcriptions.create.return_value = MagicMock(text=text)
    return mock


def test_transcribe_audio_returns_string():
    with patch("app.voice.stt._client", return_value=make_mock_client("What is AI?")):
        result = transcribe_audio(b"fake audio bytes", "question.webm")

    assert isinstance(result, str)
    assert result == "What is AI?"


def test_transcribe_audio_passes_file_bytes():
    mock_client = make_mock_client()
    with patch("app.voice.stt._client", return_value=mock_client):
        transcribe_audio(b"audio", "q.mp3")

    mock_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "whisper-1"


def test_transcribe_audio_empty_bytes_raises():
    with pytest.raises(ValueError, match="empty"):
        transcribe_audio(b"", "q.webm")
