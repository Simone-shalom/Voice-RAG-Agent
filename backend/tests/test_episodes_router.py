import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import uuid
from unittest.mock import MagicMock


def make_episode_row(filename="ep.mp3", chunk_count=5, source_url=None, summary=None, chapters=None):
    row = MagicMock()
    row.id = uuid.uuid4()
    row.filename = filename
    row.source_url = source_url
    row.chunk_count = chunk_count
    row.summary = summary
    row.chapters = chapters
    return row


def make_chunk_row(start_ts=0.0, end_ts=10.0, text="hello", speaker_label=None):
    row = MagicMock()
    row.id = uuid.uuid4()
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.speaker_label = speaker_label
    return row


def test_list_episodes_returns_list(client, mock_db):
    ep = make_episode_row()
    mock_db.execute.return_value.fetchall.return_value = [ep]
    response = client.get("/episodes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["filename"] == "ep.mp3"
    assert data[0]["chunk_count"] == 5


def test_list_episodes_empty(client, mock_db):
    mock_db.execute.return_value.fetchall.return_value = []
    response = client.get("/episodes")
    assert response.status_code == 200
    assert response.json() == []


def test_get_episode_returns_404_when_missing(client, mock_db):
    mock_db.execute.return_value.fetchone.return_value = None
    response = client.get(f"/episodes/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_episode_returns_episode(client, mock_db):
    ep = make_episode_row(filename="test.mp3", chunk_count=3)
    mock_db.execute.return_value.fetchone.return_value = ep
    response = client.get(f"/episodes/{ep.id}")
    assert response.status_code == 200
    assert response.json()["filename"] == "test.mp3"
    assert response.json()["chunk_count"] == 3


def test_get_episode_chunks_returns_list(client, mock_db):
    ep_id = str(uuid.uuid4())
    chunks = [make_chunk_row(i * 10.0, i * 10.0 + 10.0, f"text {i}") for i in range(3)]
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{ep_id}/chunks")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["text"] == "text 0"
    assert data[0]["start_ts"] == 0.0
    assert data[2]["text"] == "text 2"


def test_get_episode_includes_summary_and_chapters(client, mock_db):
    ep = make_episode_row(
        filename="test.mp3", chunk_count=3,
        summary="A summary.", chapters=[{"start_ts": 0.0, "title": "Intro"}],
    )
    mock_db.execute.return_value.fetchone.return_value = ep
    response = client.get(f"/episodes/{ep.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "A summary."
    assert body["chapters"] == [{"start_ts": 0.0, "title": "Intro"}]


def test_get_episode_summary_and_chapters_default_to_null(client, mock_db):
    ep = make_episode_row()  # summary/chapters default to None
    mock_db.execute.return_value.fetchone.return_value = ep
    response = client.get(f"/episodes/{ep.id}")
    body = response.json()
    assert body["summary"] is None
    assert body["chapters"] is None


def test_list_episodes_includes_summary_and_chapters(client, mock_db):
    ep = make_episode_row(summary="A summary.", chapters=[{"start_ts": 0.0, "title": "Intro"}])
    mock_db.execute.return_value.fetchall.return_value = [ep]
    response = client.get("/episodes")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["summary"] == "A summary."


def test_get_episode_chunks_includes_speaker_label(client, mock_db):
    ep_id = str(uuid.uuid4())
    chunks = [make_chunk_row(speaker_label="Speaker 0")]
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{ep_id}/chunks")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["speaker_label"] == "Speaker 0"


def test_get_episode_chunks_speaker_label_null_when_absent(client, mock_db):
    ep_id = str(uuid.uuid4())
    chunks = [make_chunk_row()]  # speaker_label defaults to None
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{ep_id}/chunks")
    body = response.json()
    assert body[0]["speaker_label"] is None
