from io import BytesIO
from unittest.mock import MagicMock, patch


def make_mock_episode(chunk_count=2):
    ep = MagicMock()
    ep.id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    ep.filename = "test.mp3"
    ep.chunks = [MagicMock()] * chunk_count
    return ep


def test_upload_returns_episode_id(client):
    mock_ep = make_mock_episode(chunk_count=3)

    with patch("app.ingest.router.ingest_audio", return_value=mock_ep):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(mock_ep.id)
    assert body["filename"] == "test.mp3"
    assert body["chunk_count"] == 3


def test_ingest_url_returns_episode(client):
    mock_ep = make_mock_episode(chunk_count=5)

    with patch("app.ingest.router.ingest_from_url", return_value=mock_ep):
        response = client.post("/ingest/url", json={"url": "http://example.com/audio.mp3"})

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 5


def test_upload_missing_file_returns_422(client):
    response = client.post("/ingest/upload")
    assert response.status_code == 422


def test_ingest_url_missing_url_returns_422(client):
    response = client.post("/ingest/url", json={})
    assert response.status_code == 422


def test_upload_returns_400_on_ingest_failure(client):
    with patch("app.ingest.router.ingest_audio", side_effect=RuntimeError("transcription failed")):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 400
    assert "transcription failed" in response.json()["detail"]


def test_upload_returns_400_on_tempfile_write_failure(client):
    with patch("app.ingest.router.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 400
    assert "disk full" in response.json()["detail"]


def test_ingest_url_returns_400_on_ingest_failure(client):
    with patch("app.ingest.router.ingest_from_url", side_effect=RuntimeError("download failed")):
        response = client.post("/ingest/url", json={"url": "http://example.com/audio.mp3"})

    assert response.status_code == 400
    assert "download failed" in response.json()["detail"]
