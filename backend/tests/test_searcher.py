import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.search.searcher import semantic_search


def make_mock_row(chunk_id="c1", episode_id="e1", start_ts=1.0, end_ts=5.0, text="Hello", similarity=0.9):
    row = MagicMock()
    row.id = chunk_id
    row.episode_id = episode_id
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.similarity = similarity
    return row


def test_semantic_search_returns_chunk_dicts():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        make_mock_row(),
        make_mock_row(chunk_id="c2", similarity=0.8),
    ]

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        results = semantic_search("test query", limit=10, db=db)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity"] == 0.9
    assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity"}


def test_semantic_search_passes_limit_to_query():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        semantic_search("query", limit=5, db=db)

    _, params = db.execute.call_args[0]
    assert params["limit"] == 5
