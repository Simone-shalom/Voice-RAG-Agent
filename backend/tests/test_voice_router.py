import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")

from unittest.mock import patch


def test_stt_returns_text(client):
    with patch("app.voice.router.transcribe_audio", return_value="What is AI?"):
        response = client.post(
            "/voice/stt",
            files={"audio": ("q.webm", b"fake audio", "audio/webm")},
        )
    assert response.status_code == 200
    assert response.json()["text"] == "What is AI?"


def test_stt_empty_file_returns_400(client):
    with patch("app.voice.router.transcribe_audio", side_effect=ValueError("audio bytes are empty")):
        response = client.post(
            "/voice/stt",
            files={"audio": ("q.webm", b"", "audio/webm")},
        )
    assert response.status_code == 400


def test_tts_returns_audio(client):
    with patch("app.voice.router.synthesise", return_value=b"mp3bytes"):
        response = client.post("/voice/tts", json={"text": "Hello world"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"mp3bytes"


def test_tts_empty_text_returns_400(client):
    with patch("app.voice.router.synthesise", side_effect=ValueError("text is empty")):
        response = client.post("/voice/tts", json={"text": ""})
    assert response.status_code == 400


def test_tts_missing_key_returns_400(client):
    with patch("app.voice.router.synthesise", side_effect=RuntimeError("ELEVENLABS_API_KEY not set")):
        response = client.post("/voice/tts", json={"text": "hello"})
    assert response.status_code == 400
    assert "ELEVENLABS_API_KEY" in response.json()["detail"]
