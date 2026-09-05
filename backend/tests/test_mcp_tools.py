import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mcp-server"))

os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from unittest.mock import MagicMock, patch
from tools import (
    search_knowledge_base,
    get_episode_summary,
    find_mentions,
    format_chunks_for_mcp,
)


CHUNKS = [
    {
        "chunk_id": "c1",
        "episode_id": "ep-uuid-1",
        "start_ts": 10.0,
        "end_ts": 25.5,
        "text": "RAG stands for retrieval augmented generation",
        "similarity": 0.95,
    },
    {
        "chunk_id": "c2",
        "episode_id": "ep-uuid-2",
        "start_ts": 100.0,
        "end_ts": 115.0,
        "text": "Chunking strategy affects recall significantly",
        "similarity": 0.82,
    },
]


def make_response(data, status=200):
    mock = MagicMock()
    mock.status_code = status
    mock.raise_for_status = MagicMock()
    mock.json.return_value = data
    return mock


def test_search_knowledge_base_calls_search_endpoint():
    with patch("tools.httpx.get", return_value=make_response(CHUNKS)) as mock_get:
        result = search_knowledge_base("What is RAG?")
    call = mock_get.call_args
    assert "/search" in call[0][0]
    assert call[1]["params"]["q"] == "What is RAG?"
    assert call[1]["params"]["mode"] == "hybrid"


def test_search_knowledge_base_returns_list():
    with patch("tools.httpx.get", return_value=make_response(CHUNKS)):
        result = search_knowledge_base("What is RAG?")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"


def test_search_knowledge_base_custom_limit():
    with patch("tools.httpx.get", return_value=make_response([])) as mock_get:
        search_knowledge_base("query", limit=3)
    assert mock_get.call_args[1]["params"]["limit"] == 3


def test_get_episode_summary_calls_two_endpoints():
    ep_data = {"id": "ep1", "filename": "ep.mp3", "source_url": None, "chunk_count": 50}
    chunk_data = [{"id": "c1", "start_ts": 0.0, "end_ts": 10.0, "text": "hello"}]
    with patch("tools.httpx.get") as mock_get:
        mock_get.side_effect = [
            make_response(ep_data),
            make_response(chunk_data),
        ]
        result = get_episode_summary("ep1")
    assert mock_get.call_count == 2
    assert result["episode"]["filename"] == "ep.mp3"
    assert len(result["chunks"]) == 1


def test_get_episode_summary_returns_episode_and_chunks():
    ep_data = {"id": "ep1", "filename": "test.mp3", "source_url": "http://x", "chunk_count": 10}
    chunk_data = [{"id": "c1", "start_ts": 0.0, "end_ts": 5.0, "text": "intro"}]
    with patch("tools.httpx.get") as mock_get:
        mock_get.side_effect = [make_response(ep_data), make_response(chunk_data)]
        result = get_episode_summary("ep1")
    assert result["episode"]["source_url"] == "http://x"
    assert result["chunks"][0]["text"] == "intro"


def test_find_mentions_returns_list():
    with patch("tools.httpx.get", return_value=make_response(CHUNKS)):
        result = find_mentions("fine-tuning")
    assert isinstance(result, list)
    assert len(result) == 2


def test_find_mentions_uses_hybrid_mode():
    with patch("tools.httpx.get", return_value=make_response([])) as mock_get:
        find_mentions("topic")
    assert mock_get.call_args[1]["params"]["mode"] == "hybrid"


def test_format_chunks_for_mcp_empty():
    assert format_chunks_for_mcp([]) == "No results found."


def test_format_chunks_for_mcp_includes_timestamps():
    output = format_chunks_for_mcp(CHUNKS)
    assert "10.0s" in output
    assert "25.5s" in output
    assert "RAG stands for" in output


def test_format_chunks_for_mcp_includes_episode_id():
    output = format_chunks_for_mcp(CHUNKS)
    assert "ep-uuid-1" in output
    assert "ep-uuid-2" in output


def test_format_chunks_for_mcp_numbered():
    output = format_chunks_for_mcp(CHUNKS)
    assert "[1]" in output
    assert "[2]" in output
