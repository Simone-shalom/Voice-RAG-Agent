import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.agent.tools import make_tools


def make_db():
    return MagicMock()


def test_make_tools_returns_four_tools():
    tools = make_tools(make_db())
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {
        "search_transcripts",
        "get_context_around_timestamp",
        "compare_across_episodes",
        "summarise_segment",
    }


def test_search_transcripts_returns_string():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "hello world", "similarity": 0.9}
    ]):
        tools = make_tools(db)
        search = next(t for t in tools if t.name == "search_transcripts")
        result = search.invoke({"query": "hello", "limit": 5})

    assert isinstance(result, str)
    assert "hello world" in result
    assert "0.0s" in result


def test_get_context_around_timestamp_returns_string():
    db = make_db()
    db.execute.return_value.fetchall.return_value = [
        MagicMock(id="c1", episode_id="ep1", start_ts=4.0, end_ts=9.0, text="nearby chunk")
    ]
    tools = make_tools(db)
    fn = next(t for t in tools if t.name == "get_context_around_timestamp")
    result = fn.invoke({"episode_id": "ep1", "timestamp": 5.0, "window_seconds": 10.0})

    assert isinstance(result, str)
    assert "nearby chunk" in result


def test_compare_across_episodes_returns_string():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "ep1 content", "similarity": 0.9},
        {"chunk_id": "c2", "episode_id": "ep2", "start_ts": 10.0, "end_ts": 15.0,
         "text": "ep2 content", "similarity": 0.8},
    ]):
        tools = make_tools(db)
        fn = next(t for t in tools if t.name == "compare_across_episodes")
        result = fn.invoke({"query": "topic", "episode_ids": "ep1,ep2", "limit_per_episode": 3})

    assert isinstance(result, str)
    assert "ep1" in result
    assert "ep2" in result


def test_summarise_segment_returns_string():
    tools = make_tools(make_db())
    fn = next(t for t in tools if t.name == "summarise_segment")
    result = fn.invoke({"text": "This is a long passage about AI that needs summarising."})

    assert isinstance(result, str)
    assert "AI" in result


def test_search_transcripts_appends_to_sources_sink():
    db = make_db()
    sink: list[dict] = []
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "hello world", "similarity": 0.9}
    ]):
        tools = make_tools(db, sources_sink=sink)
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    assert sink == [{"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
                      "text": "hello world", "similarity": 0.9}]


def test_search_transcripts_uses_requested_mode():
    db = make_db()
    with patch("app.agent.tools.semantic_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "semantic hit", "similarity": 0.5}
    ]) as mock_semantic, \
         patch("app.agent.tools.hybrid_search") as mock_hybrid:
        tools = make_tools(db, mode="semantic")
        search = next(t for t in tools if t.name == "search_transcripts")
        result = search.invoke({"query": "hello", "limit": 5})

    mock_semantic.assert_called_once()
    mock_hybrid.assert_not_called()
    assert "semantic hit" in result


def test_unrecognised_mode_falls_back_to_hybrid():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[]) as mock_hybrid:
        tools = make_tools(db, mode="not-a-real-mode")
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    mock_hybrid.assert_called_once()


def test_get_context_around_timestamp_appends_to_sources_sink():
    db = make_db()
    db.execute.return_value.fetchall.return_value = [
        MagicMock(id="c1", episode_id="ep1", start_ts=4.0, end_ts=9.0, text="nearby chunk")
    ]
    sink: list[dict] = []
    tools = make_tools(db, sources_sink=sink)
    fn = next(t for t in tools if t.name == "get_context_around_timestamp")
    fn.invoke({"episode_id": "ep1", "timestamp": 5.0, "window_seconds": 10.0})

    assert sink == [{"episode_id": "ep1", "start_ts": 4.0, "end_ts": 9.0, "text": "nearby chunk"}]


def test_compare_across_episodes_appends_to_sources_sink():
    db = make_db()
    sink: list[dict] = []
    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
         "text": "ep1 content", "similarity": 0.9},
    ]):
        tools = make_tools(db, sources_sink=sink)
        fn = next(t for t in tools if t.name == "compare_across_episodes")
        fn.invoke({"query": "topic", "episode_ids": "ep1", "limit_per_episode": 3})

    assert len(sink) == 1
    assert sink[0]["episode_id"] == "ep1"


def test_hybrid_rerank_falls_back_to_hybrid_without_cohere_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[]) as mock_hybrid, \
         patch("app.agent.tools.hybrid_rerank_search") as mock_rerank:
        tools = make_tools(db, mode="hybrid+rerank")
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    mock_hybrid.assert_called_once()
    mock_rerank.assert_not_called()


def test_hybrid_rerank_used_when_cohere_key_present(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    db = make_db()
    with patch("app.agent.tools.hybrid_search") as mock_hybrid, \
         patch("app.agent.tools.hybrid_rerank_search", return_value=[]) as mock_rerank:
        tools = make_tools(db, mode="hybrid+rerank")
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 5})

    mock_rerank.assert_called_once()
    mock_hybrid.assert_not_called()


def test_search_transcripts_clamps_excessive_limit():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[]) as mock_hybrid:
        tools = make_tools(db)
        search = next(t for t in tools if t.name == "search_transcripts")
        search.invoke({"query": "hello", "limit": 9999})

    assert mock_hybrid.call_args.kwargs["limit"] == 20


def test_compare_across_episodes_clamps_excessive_limit():
    db = make_db()
    with patch("app.agent.tools.hybrid_search", return_value=[]) as mock_hybrid:
        tools = make_tools(db)
        fn = next(t for t in tools if t.name == "compare_across_episodes")
        fn.invoke({"query": "topic", "episode_ids": "ep1", "limit_per_episode": 9999})

    assert mock_hybrid.call_args.kwargs["limit"] == 20
