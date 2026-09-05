import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch

import pytest

from app.ingest.service import ingest_audio


def make_mock_db():
    db = MagicMock()
    return db


def test_ingest_audio_creates_episode_and_chunks(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    mock_segments = [
        {"text": "Hello world.", "start": 0.0, "end": 2.0},
        {"text": "Another sentence.", "start": 3.0, "end": 5.0},  # pause > 0.5s -> 2 chunks
    ]
    mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

    with patch("app.ingest.service.transcribe", return_value=mock_segments) as mock_t, \
         patch("app.ingest.service.embed_texts", return_value=mock_embeddings) as mock_e:
        ingest_audio(audio_file, "test.mp3", None, db)

    mock_t.assert_called_once_with(audio_file)
    mock_e.assert_called_once()
    assert db.add.call_count == 3  # 1 episode + 2 chunks
    db.commit.assert_called_once()


def test_ingest_audio_sets_episode_filename_and_source(tmp_path):
    audio_file = tmp_path / "my_podcast.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]):
        ingest_audio(audio_file, "my_podcast.mp3", "http://example.com/audio.mp3", db)

    added_episode = db.add.call_args_list[0][0][0]
    assert added_episode.filename == "my_podcast.mp3"
    assert added_episode.source_url == "http://example.com/audio.mp3"


def test_ingest_audio_does_not_commit_if_transcribe_fails(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", side_effect=RuntimeError("transcription failed")):
        with pytest.raises(RuntimeError):
            ingest_audio(audio_file, "test.mp3", None, db)

    db.add.assert_not_called()
    db.commit.assert_not_called()
