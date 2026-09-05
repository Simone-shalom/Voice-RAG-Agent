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
