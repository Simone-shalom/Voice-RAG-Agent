import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock
from app.search.bm25 import bm25_search


def make_mock_row(chunk_id="c1", episode_id="e1", start_ts=0.0, end_ts=5.0, text="Hello", bm25_score=0.5, speaker_label=None):
    row = MagicMock()
    row.id = chunk_id
    row.episode_id = episode_id
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.bm25_score = bm25_score
    row.speaker_label = speaker_label
    return row


def test_bm25_search_returns_chunk_dicts():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        make_mock_row(),
        make_mock_row(chunk_id="c2", bm25_score=0.3),
    ]

    results = bm25_search("hello world", limit=10, db=db)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity"] == 0.5
    assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity", "speaker_label"}


def test_bm25_search_returns_speaker_label_when_present():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [make_mock_row(speaker_label="Speaker 1")]

    results = bm25_search("hello world", limit=10, db=db)

    assert results[0]["speaker_label"] == "Speaker 1"


def test_bm25_search_passes_limit():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    bm25_search("query", limit=7, db=db)

    _, params = db.execute.call_args[0]
    assert params["limit"] == 7


def test_bm25_search_empty_results():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    results = bm25_search("no match", limit=10, db=db)

    assert results == []
