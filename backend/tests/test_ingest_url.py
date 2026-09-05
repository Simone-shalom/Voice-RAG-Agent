import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.ingest.service import MAX_DOWNLOAD_BYTES, ingest_from_url


def _fake_episode() -> MagicMock:
    ep = MagicMock()
    ep.id = uuid.uuid4()
    ep.filename = "ep1.mp3"
    ep.chunks = []
    return ep


def _mock_stream_response(chunks: list[bytes]):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_bytes.return_value = iter(chunks)
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    return context


def _mock_client(chunks: list[bytes]) -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.stream.return_value = _mock_stream_response(chunks)
    return client


def test_ingest_from_url_validates_url_before_fetching():
    with patch("app.ingest.service.assert_safe_url", side_effect=ValueError("blocked")) as mock_assert, \
         patch("app.ingest.service.httpx.Client") as mock_client_cls:
        with pytest.raises(ValueError, match="blocked"):
            ingest_from_url("http://169.254.169.254/", db=MagicMock())

    mock_assert.assert_called_once_with("http://169.254.169.254/")
    mock_client_cls.assert_not_called()


def test_ingest_from_url_downloads_and_ingests():
    fake_episode = _fake_episode()
    mock_client = _mock_client([b"abc", b"def"])

    with patch("app.ingest.service.assert_safe_url"), \
         patch("app.ingest.service.httpx.Client", return_value=mock_client), \
         patch("app.ingest.service.ingest_audio", return_value=fake_episode) as mock_ingest_audio:
        result = ingest_from_url("https://example.com/ep1.mp3", db=MagicMock())

    assert result is fake_episode
    args, _kwargs = mock_ingest_audio.call_args
    assert args[1] == "ep1.mp3"
    assert args[2] == "https://example.com/ep1.mp3"


def test_ingest_from_url_rejects_oversized_download():
    oversized_chunk = b"x" * (MAX_DOWNLOAD_BYTES + 1)
    mock_client = _mock_client([oversized_chunk])

    with patch("app.ingest.service.assert_safe_url"), \
         patch("app.ingest.service.httpx.Client", return_value=mock_client):
        with pytest.raises(ValueError, match="exceeds"):
            ingest_from_url("https://example.com/big.mp3", db=MagicMock())


def test_ingest_from_url_passes_client_url_to_stream():
    fake_episode = _fake_episode()
    mock_client = _mock_client([b"data"])

    with patch("app.ingest.service.assert_safe_url"), \
         patch("app.ingest.service.httpx.Client", return_value=mock_client), \
         patch("app.ingest.service.ingest_audio", return_value=fake_episode):
        ingest_from_url("https://example.com/ep1.mp3", db=MagicMock())

    mock_client.stream.assert_called_once_with("GET", "https://example.com/ep1.mp3")
