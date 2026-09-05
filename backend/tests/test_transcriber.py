import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.ingest.transcriber import transcribe


def make_mock_response():
    seg1 = MagicMock(text="Hello world.", start=0.0, end=2.0)
    seg2 = MagicMock(text="This is a test.", start=2.5, end=4.0)
    response = MagicMock()
    response.segments = [seg1, seg2]
    return response


def make_mock_client():
    mock = MagicMock()
    mock.audio.transcriptions.create.return_value = make_mock_response()
    return mock


def test_transcribe_returns_segment_list(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("app.ingest.transcriber._client", return_value=make_mock_client()):
        result = transcribe(audio_file)

    assert len(result) == 2
    assert result[0] == {"text": "Hello world.", "start": 0.0, "end": 2.0}


def test_transcribe_requests_word_and_segment_timestamps(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    mock_client = make_mock_client()
    with patch("app.ingest.transcriber._client", return_value=mock_client):
        transcribe(audio_file)

    call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["timestamp_granularities"] == ["word", "segment"]
    assert call_kwargs["response_format"] == "verbose_json"
