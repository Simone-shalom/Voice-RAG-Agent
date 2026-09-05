import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
import pytest
from app.search.reranker import rerank


def make_chunks(n=3):
    return [
        {"chunk_id": chr(ord("a") + i), "episode_id": "ep1",
         "start_ts": float(i), "end_ts": float(i + 5),
         "text": f"content {chr(ord('a') + i)}", "similarity": 0.5}
        for i in range(n)
    ]


def make_mock_client(reranked_indices: list[int]):
    mock = MagicMock()
    mock.rerank.return_value.results = [
        MagicMock(index=i, relevance_score=1.0 - rank * 0.1)
        for rank, i in enumerate(reranked_indices)
    ]
    return mock


def test_rerank_reorders_chunks():
    chunks = make_chunks(3)  # a, b, c
    mock_client = make_mock_client([2, 0, 1])  # Cohere says c > a > b

    with patch("app.search.reranker._client", return_value=mock_client):
        result = rerank("query", chunks, top_n=3)

    assert result[0]["chunk_id"] == "c"
    assert result[1]["chunk_id"] == "a"
    assert result[2]["chunk_id"] == "b"


def test_rerank_sets_similarity_to_relevance_score():
    chunks = make_chunks(1)
    mock_client = make_mock_client([0])
    mock_client.rerank.return_value.results[0].relevance_score = 0.92

    with patch("app.search.reranker._client", return_value=mock_client):
        result = rerank("query", chunks, top_n=1)

    assert result[0]["similarity"] == pytest.approx(0.92)


def test_rerank_empty_chunks_returns_empty():
    result = rerank("query", [], top_n=5)
    assert result == []


def test_rerank_raises_if_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        from app.search import reranker
        reranker._client.cache_clear()
        with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
            rerank("query", make_chunks(1), top_n=1)
    reranker._client.cache_clear()
