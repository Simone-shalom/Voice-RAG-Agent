import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest.diarizer import assign_speaker_labels, diarize_audio, merge_into_speaker_segments


def test_diarize_audio_raises_without_api_key(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            diarize_audio(audio_file)


def test_diarize_audio_parses_word_level_speakers(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": 0},
                                {"word": "there", "start": 0.5, "end": 1.0, "speaker": 0},
                                {"word": "hi", "start": 1.2, "end": 1.5, "speaker": 1},
                            ]
                        }
                    ]
                }
            ]
        }
    }

    with patch("app.ingest.diarizer.httpx.post", return_value=mock_response) as mock_post, \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
        words = diarize_audio(audio_file)

    assert words == [
        {"start": 0.0, "end": 0.5, "speaker": 0},
        {"start": 0.5, "end": 1.0, "speaker": 0},
        {"start": 1.2, "end": 1.5, "speaker": 1},
    ]
    call_kwargs = mock_post.call_args
    assert "diarize=true" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["Authorization"] == "Token test-key"


def test_merge_into_speaker_segments_collapses_consecutive_same_speaker():
    words = [
        {"start": 0.0, "end": 0.5, "speaker": 0},
        {"start": 0.5, "end": 1.0, "speaker": 0},
        {"start": 1.2, "end": 1.5, "speaker": 1},
        {"start": 1.5, "end": 2.0, "speaker": 1},
        {"start": 2.1, "end": 2.5, "speaker": 0},
    ]
    segments = merge_into_speaker_segments(words)
    assert segments == [
        {"start_ts": 0.0, "end_ts": 1.0, "speaker": 0},
        {"start_ts": 1.2, "end_ts": 2.0, "speaker": 1},
        {"start_ts": 2.1, "end_ts": 2.5, "speaker": 0},
    ]


def test_merge_into_speaker_segments_empty_input():
    assert merge_into_speaker_segments([]) == []


def test_assign_speaker_labels_picks_majority_overlap_speaker():
    chunks = [{"start_ts": 0.0, "end_ts": 2.0, "text": "hello there"}]
    # Speaker 0 covers [0.0, 1.5] = 1.5s of this 2.0s chunk = 75% > 50% threshold
    words = [
        {"start": 0.0, "end": 1.5, "speaker": 0},
        {"start": 1.5, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == ["Speaker 0"]


def test_assign_speaker_labels_returns_none_below_threshold():
    chunks = [{"start_ts": 0.0, "end_ts": 2.0, "text": "hello there"}]
    # Speaker 0 covers exactly 1.0s of this 2.0s chunk = 50%, NOT strictly > 50%
    words = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == [None]


def test_assign_speaker_labels_returns_none_with_no_overlapping_words():
    chunks = [{"start_ts": 10.0, "end_ts": 12.0, "text": "hello"}]
    words = [{"start": 0.0, "end": 1.0, "speaker": 0}]
    labels = assign_speaker_labels(chunks, words)
    assert labels == [None]


def test_assign_speaker_labels_handles_multiple_chunks_in_order():
    chunks = [
        {"start_ts": 0.0, "end_ts": 1.0, "text": "a"},
        {"start_ts": 1.0, "end_ts": 2.0, "text": "b"},
    ]
    words = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == ["Speaker 0", "Speaker 1"]
