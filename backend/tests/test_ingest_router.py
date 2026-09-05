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
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
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
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 400
    assert "transcription failed" in response.json()["detail"]


def test_upload_returns_500_and_hides_detail_on_unexpected_failure(client):
    with patch("app.ingest.router.tempfile.NamedTemporaryFile", side_effect=OSError("disk full at /var/lib/secret")):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 500
    assert "disk full" not in response.json()["detail"]
    assert "secret" not in response.json()["detail"]


def test_ingest_url_returns_400_on_ingest_failure(client):
    with patch("app.ingest.router.ingest_from_url", side_effect=RuntimeError("download failed")):
        response = client.post("/ingest/url", json={"url": "http://example.com/audio.mp3"})

    assert response.status_code == 400
    assert "download failed" in response.json()["detail"]


def test_ingest_rss_returns_episodes_and_errors(client, mock_db):
    fake_episode = MagicMock()
    fake_episode.id = "11111111-1111-1111-1111-111111111111"
    fake_episode.filename = "ep1.mp3"
    fake_episode.chunks = [MagicMock()]

    with patch(
        "app.ingest.router.ingest_from_rss",
        return_value={"episodes": [fake_episode], "errors": [{"title": "Bad", "error": "boom"}]},
    ):
        res = client.post("/ingest/rss", json={"url": "https://example.com/feed.xml", "limit": 5})

    assert res.status_code == 200
    body = res.json()
    assert len(body["episodes"]) == 1
    assert body["episodes"][0]["filename"] == "ep1.mp3"
    assert len(body["errors"]) == 1
    assert body["errors"][0]["title"] == "Bad"


def test_ingest_rss_defaults_limit_to_5(client, mock_db):
    with patch(
        "app.ingest.router.ingest_from_rss",
        return_value={"episodes": [], "errors": []},
    ) as mock_ingest:
        client.post("/ingest/rss", json={"url": "https://example.com/feed.xml"})

    mock_ingest.assert_called_once()
    assert mock_ingest.call_args.args[0] == "https://example.com/feed.xml"
    assert mock_ingest.call_args.kwargs.get("limit") == 5


def test_ingest_rss_batch_level_failure_returns_400(client, mock_db):
    with patch(
        "app.ingest.router.ingest_from_rss",
        side_effect=RuntimeError("feed unreachable"),
    ):
        res = client.post("/ingest/rss", json={"url": "https://example.com/bad-feed.xml"})

    assert res.status_code == 400
    assert "feed unreachable" in res.json()["detail"]


def test_ingest_rss_forwards_non_default_limit(client, mock_db):
    with patch(
        "app.ingest.router.ingest_from_rss",
        return_value={"episodes": [], "errors": []},
    ) as mock_ingest:
        client.post("/ingest/rss", json={"url": "https://example.com/feed.xml", "limit": 3})

    mock_ingest.assert_called_once()
    assert mock_ingest.call_args.kwargs.get("limit") == 3


def test_upload_rejects_non_audio_content(client):
    response = client.post(
        "/ingest/upload",
        files={"file": ("not_audio.bin", BytesIO(b"just some plain text bytes"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "does not look like" in response.json()["detail"]


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr("app.ingest.router.MAX_UPLOAD_BYTES", 10)
    response = client.post(
        "/ingest/upload",
        files={"file": ("big.mp3", BytesIO(b"ID3" + b"\x00" * 20), "audio/mpeg")},
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"]


def test_ingest_rss_rejects_limit_above_20(client):
    response = client.post("/ingest/rss", json={"url": "https://example.com/feed.xml", "limit": 21})
    assert response.status_code == 422


def test_ingest_rss_rejects_limit_below_1(client):
    response = client.post("/ingest/rss", json={"url": "https://example.com/feed.xml", "limit": 0})
    assert response.status_code == 422


def test_ingest_url_returns_500_and_hides_detail_on_unexpected_failure(client):
    with patch("app.ingest.router.ingest_from_url", side_effect=OSError("leaked internal path /srv/data")):
        response = client.post("/ingest/url", json={"url": "http://example.com/audio.mp3"})

    assert response.status_code == 500
    assert "leaked internal path" not in response.json()["detail"]


def test_ingest_rss_returns_500_and_hides_detail_on_unexpected_failure(client, mock_db):
    with patch("app.ingest.router.ingest_from_rss", side_effect=OSError("leaked internal path")):
        response = client.post("/ingest/rss", json={"url": "https://example.com/feed.xml"})

    assert response.status_code == 500
    assert "leaked internal path" not in response.json()["detail"]


def test_ingest_upload_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    response = client.post(
        "/ingest/upload",
        files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
    )
    assert response.status_code == 401


def test_ingest_upload_accepts_correct_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    mock_ep = make_mock_episode(chunk_count=1)
    with patch("app.ingest.router.ingest_audio", return_value=mock_ep):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
            headers={"X-API-Key": "secret123"},
        )
    assert response.status_code == 200


def test_ingest_rss_rate_limited_when_enabled(client, monkeypatch):
    from app.core.rate_limit import limiter as rate_limiter
    monkeypatch.setattr(rate_limiter, "enabled", True)
    with patch("app.ingest.router.ingest_from_rss", return_value={"episodes": [], "errors": []}):
        statuses = [
            client.post("/ingest/rss", json={"url": "https://example.com/feed.xml"}).status_code
            for _ in range(6)  # limit is 5/minute
        ]
    assert statuses[-1] == 429


def test_ingest_url_returns_400_on_httpx_error(client):
    import httpx as httpx_module
    with patch("app.ingest.router.ingest_from_url", side_effect=httpx_module.ConnectError("connection refused")):
        response = client.post("/ingest/url", json={"url": "http://example.com/audio.mp3"})
    assert response.status_code == 400
    assert response.json()["detail"] == "could not fetch the URL"


def test_upload_returns_400_on_openai_api_error(client):
    from openai import APIError as OpenAIAPIError
    fake_request = MagicMock()
    with patch(
        "app.ingest.router.ingest_audio",
        side_effect=OpenAIAPIError("rate limited", request=fake_request, body=None),
    ):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "audio transcription failed"
