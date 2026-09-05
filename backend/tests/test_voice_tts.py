import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
import pytest
from app.voice.tts import synthesise


def make_mock_response(content=b"fake_mp3", status=200):
    mock = MagicMock()
    mock.content = content
    mock.status_code = status
    mock.raise_for_status = MagicMock()
    return mock


def test_synthesise_returns_bytes():
    with patch("app.voice.tts.httpx") as mock_httpx:
        mock_httpx.post.return_value = make_mock_response(b"mp3data")
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            result = synthesise("Hello world")

    assert isinstance(result, bytes)
    assert result == b"mp3data"


def test_synthesise_raises_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
            synthesise("test")


def test_synthesise_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        synthesise("")


def test_synthesise_calls_correct_endpoint():
    with patch("app.voice.tts.httpx") as mock_httpx:
        mock_httpx.post.return_value = make_mock_response()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "key123"}):
            synthesise("hello")

    call_args = mock_httpx.post.call_args
    assert "text-to-speech" in call_args[0][0]
    assert call_args[1]["headers"]["xi-api-key"] == "key123"
