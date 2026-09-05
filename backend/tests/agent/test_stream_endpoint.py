import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import patch


async def _fake_stream(*args, **kwargs):
    yield 'data: {"type": "token", "text": "Hello"}\n\n'
    yield 'data: {"type": "done"}\n\n'


def test_stream_endpoint_returns_200(client):
    with patch("app.agent.router.stream_agent_response", side_effect=_fake_stream):
        response = client.post(
            "/agent/chat/stream",
            json={"message": "hello", "thread_id": "t1"},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_stream_endpoint_body_contains_done(client):
    with patch("app.agent.router.stream_agent_response", side_effect=_fake_stream):
        response = client.post(
            "/agent/chat/stream",
            json={"message": "hello", "thread_id": "t1"},
        )
    assert b'"type": "done"' in response.content
