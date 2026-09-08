import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import postgresql

from app.search.searcher import semantic_search, hybrid_search, hybrid_rerank_search


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


def test_semantic_search_query_actually_binds_emb_param():
    """
    Regression test: `:emb::vector` looks like a bind param followed by a
    Postgres cast, but SQLAlchemy's text() bind-param regex has a negative
    lookahead on a trailing `:` (to avoid colliding with `::` casts) and
    silently refuses to treat `:emb` as a param at all when compiled — the
    mock-based tests above never caught this because they only check the
    Python-level dict passed to db.execute(), not what the driver actually
    receives. Compiling against the real postgres dialect catches it.
    """
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        semantic_search("query", limit=5, db=db)

    query, _ = db.execute.call_args[0]
    compiled_params = query.compile(dialect=postgresql.dialect()).params
    assert "emb" in compiled_params
    assert "limit" in compiled_params


def _make_chunk(cid, sim=0.5):
    return {"chunk_id": cid, "episode_id": "e1", "start_ts": 0.0, "end_ts": 5.0,
            "text": f"text {cid}", "similarity": sim}


def test_hybrid_search_fuses_semantic_and_bm25():
    sem = [_make_chunk("a", 0.9), _make_chunk("b", 0.7)]
    bm25 = [_make_chunk("b", 0.8), _make_chunk("c", 0.6)]
    db = MagicMock()

    with patch("app.search.searcher.semantic_search", return_value=sem) as m_sem, \
         patch("app.search.searcher.bm25_search", return_value=bm25) as m_bm25:
        result = hybrid_search("test query", limit=3, db=db)

    assert m_sem.call_args.kwargs["limit"] == 6
    assert m_bm25.call_args.kwargs["limit"] == 6
    assert result[0]["chunk_id"] == "b"
    assert {r["chunk_id"] for r in result} == {"a", "b", "c"}
    assert "similarity" in result[0]


def test_hybrid_search_returns_at_most_limit():
    sem = [_make_chunk(str(i)) for i in range(20)]
    db = MagicMock()

    with patch("app.search.searcher.semantic_search", return_value=sem), \
         patch("app.search.searcher.bm25_search", return_value=[]):
        result = hybrid_search("q", limit=5, db=db)

    assert len(result) <= 5


def test_hybrid_rerank_search_calls_reranker():
    fused = [_make_chunk("x")]
    reranked = [{**fused[0], "similarity": 0.95}]
    db = MagicMock()

    with patch("app.search.searcher.hybrid_search", return_value=fused), \
         patch("app.search.searcher.rerank", return_value=reranked):
        result = hybrid_rerank_search("query", limit=5, db=db)

    assert result[0]["similarity"] == 0.95
