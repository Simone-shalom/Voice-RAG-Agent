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

    # diarize_audio/summarize_episode are mocked to raise here (simulating unset
    # DEEPGRAM_API_KEY/ANTHROPIC_API_KEY) rather than left unmocked: this test's
    # own process may run after another test module that has already permanently
    # set ANTHROPIC_API_KEY via os.environ.setdefault (e.g. test_agent_graph.py),
    # so leaving summarize_episode unmocked would let it slip past its internal
    # "key unset" guard and make a REAL, unmocked network call to Anthropic —
    # exactly the kind of cross-test pollution this project's test suite is
    # designed to avoid (see CLAUDE.md: "244 tests, all mocked"). That real call
    # was observed to poison summarizer._client()'s lru_cache for the rest of the
    # test session, breaking an unrelated test in test_summarizer.py.
    with patch("app.ingest.service.transcribe", return_value=mock_segments) as mock_t, \
         patch("app.ingest.service.embed_texts", return_value=mock_embeddings) as mock_e, \
         patch("app.ingest.service.diarize_audio", side_effect=RuntimeError("DEEPGRAM_API_KEY not set")), \
         patch("app.ingest.service.summarize_episode", side_effect=RuntimeError("ANTHROPIC_API_KEY not set")):
        ingest_audio(audio_file, "test.mp3", None, db)

    mock_t.assert_called_once_with(audio_file)
    mock_e.assert_called_once()
    assert db.add.call_count == 3  # 1 episode + 2 chunks
    db.commit.assert_called_once()


def test_ingest_audio_sets_speaker_label_and_summary_on_success(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    mock_segments = [{"text": "Hello world.", "start": 0.0, "end": 2.0}]
    mock_words = [{"start": 0.0, "end": 2.0, "speaker": 0}]
    mock_summary_result = {
        "summary": "A short greeting.",
        "chapters": [{"start_ts": 0.0, "title": "Greeting"}],
        "speaker_names": {"Speaker 0": "Alex"},
    }

    with patch("app.ingest.service.transcribe", return_value=mock_segments), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=mock_words), \
         patch("app.ingest.service.summarize_episode", return_value=mock_summary_result) as mock_summarize:
        ingest_audio(audio_file, "test.mp3", None, db)

    added_episode = db.add.call_args_list[0][0][0]
    added_chunk = db.add.call_args_list[1][0][0]
    assert added_episode.summary == "A short greeting."
    assert added_episode.chapters == [{"start_ts": 0.0, "title": "Greeting"}]
    # Speaker 0's overlap covers the whole 2.0s chunk (100% > 50% threshold),
    # and speaker_names remaps "Speaker 0" -> "Alex" from the same summarizer call.
    assert added_chunk.speaker_label == "Alex"
    # speaker_names is only grounded if summarize_episode was actually shown
    # which chunk belongs to which diarizer speaker index — assert the chunks
    # PASSED IN already carry the diarizer's generic label, not just that the
    # final persisted Chunk.speaker_label happens to come out right.
    chunks_sent_to_summarizer = mock_summarize.call_args[0][0]
    assert chunks_sent_to_summarizer[0]["speaker_label"] == "Speaker 0"


def test_ingest_audio_survives_diarization_failure(tmp_path):
    """
    DoD requirement: a diarization outage (bad key/timeout) must leave the
    episode fully ingested and searchable with speaker_label=None, not crash
    the ingest.
    """
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", side_effect=RuntimeError("DEEPGRAM_API_KEY not set")), \
         patch("app.ingest.service.summarize_episode", return_value={"summary": "", "chapters": [], "speaker_names": {}}):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_chunk = db.add.call_args_list[1][0][0]
    assert added_chunk.speaker_label is None
    db.commit.assert_called_once()


def test_ingest_audio_survives_summarization_failure(tmp_path):
    """DoD requirement: a summarization outage must leave summary/chapters=None, not crash."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=[]), \
         patch("app.ingest.service.summarize_episode", side_effect=RuntimeError("ANTHROPIC_API_KEY not set")):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_episode = db.add.call_args_list[0][0][0]
    assert added_episode.summary is None
    assert added_episode.chapters is None
    db.commit.assert_called_once()


def test_ingest_audio_speaker_label_stays_generic_without_name_mapping(tmp_path):
    """When summarize_episode's speaker_names has no entry for a Speaker N label, keep the generic label."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}]), \
         patch("app.ingest.service.summarize_episode", return_value={"summary": "s", "chapters": [], "speaker_names": {}}):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_chunk = db.add.call_args_list[1][0][0]
    assert added_chunk.speaker_label == "Speaker 0"


def test_ingest_audio_sets_episode_filename_and_source(tmp_path):
    audio_file = tmp_path / "my_podcast.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    # See the comment in test_ingest_audio_creates_episode_and_chunks above for why
    # diarize_audio/summarize_episode must be mocked even though this test doesn't
    # otherwise care about them.
    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", side_effect=RuntimeError("DEEPGRAM_API_KEY not set")), \
         patch("app.ingest.service.summarize_episode", side_effect=RuntimeError("ANTHROPIC_API_KEY not set")):
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
