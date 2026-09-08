import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage


def test_agent_chat_returns_reply(client):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="The answer is 42.")],
        "step_count": 1,
    }

    with patch("app.agent.router.build_graph", return_value=mock_graph):
        response = client.post("/agent/chat", json={"message": "What is 6x7?", "thread_id": "t1"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "The answer is 42."
    assert body["steps"] == 1


def test_agent_chat_missing_message_returns_422(client):
    response = client.post("/agent/chat", json={"thread_id": "t1"})
    assert response.status_code == 422


def test_agent_chat_missing_anthropic_key_returns_400(client):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        response = client.post("/agent/chat", json={"message": "hi", "thread_id": "t2"})
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_chat_returns_500_and_hides_detail_on_unexpected_failure(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("app.agent.router.build_graph", side_effect=OSError("leaked internal path")):
        response = client.post("/agent/chat", json={"message": "hi", "thread_id": "t1"})

    assert response.status_code == 500
    assert "leaked internal path" not in response.json()["detail"]


def test_chat_persists_user_and_assistant_messages(client):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="The answer is 42.")],
        "step_count": 1,
    }

    with patch("app.agent.router.build_graph", return_value=mock_graph), \
         patch("app.agent.router.ensure_thread") as mock_ensure, \
         patch("app.agent.router.save_message") as mock_save:
        client.post("/agent/chat", json={"message": "What is 6x7?", "thread_id": "t1"})

    mock_ensure.assert_called_once()
    assert mock_ensure.call_args[0][1:] == ("t1", "What is 6x7?")
    assert mock_save.call_count == 2
    assert mock_save.call_args_list[0][0][1:] == ("t1", "user", "What is 6x7?", None)
    assert mock_save.call_args_list[1][0][1:] == ("t1", "assistant", "The answer is 42.", None)


def test_chat_persistence_failure_does_not_break_response(client):
    """A history-write failure must never surface as a broken chat response."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="The answer is 42.")],
        "step_count": 1,
    }

    with patch("app.agent.router.build_graph", return_value=mock_graph), \
         patch("app.agent.router.save_message", side_effect=RuntimeError("db down")):
        response = client.post("/agent/chat", json={"message": "hi", "thread_id": "t1"})

    assert response.status_code == 200
    assert response.json()["reply"] == "The answer is 42."


def test_get_threads_returns_list(client):
    fake_thread = MagicMock(id="t1", title="What is the ISS National Lab?")
    fake_thread.updated_at = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

    with patch("app.agent.router.list_threads", return_value=[fake_thread]):
        response = client.get("/agent/threads")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "t1"
    assert body[0]["title"] == "What is the ISS National Lab?"


def test_get_thread_detail_returns_messages(client):
    fake_message = MagicMock(role="user", text="hello", sources=None)
    fake_message.created_at = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    fake_thread = MagicMock(id="t1", title="hello", messages=[fake_message])

    with patch("app.agent.router.get_thread", return_value=fake_thread):
        response = client.get("/agent/threads/t1")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "t1"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["text"] == "hello"


def test_get_thread_detail_404_when_missing(client):
    with patch("app.agent.router.get_thread", return_value=None):
        response = client.get("/agent/threads/nonexistent")

    assert response.status_code == 404


def test_delete_thread_returns_204(client):
    with patch("app.agent.router.delete_thread", return_value=True) as mock_delete:
        response = client.delete("/agent/threads/t1")

    assert response.status_code == 204
    mock_delete.assert_called_once()
    assert mock_delete.call_args[0][1] == "t1"


def test_delete_thread_404_when_missing(client):
    with patch("app.agent.router.delete_thread", return_value=False):
        response = client.delete("/agent/threads/nonexistent")

    assert response.status_code == 404
