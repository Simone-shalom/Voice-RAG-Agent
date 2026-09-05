import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

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
