import uuid
from unittest.mock import MagicMock, patch

from app.ingest.service import ingest_from_rss


def _fake_episode(filename: str) -> MagicMock:
    ep = MagicMock()
    ep.id = uuid.uuid4()
    ep.filename = filename
    ep.chunks = [MagicMock(), MagicMock()]
    return ep


def test_ingest_from_rss_ingests_all_entries():
    entries = [
        {"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"},
        {"title": "Ep 2", "audio_url": "https://example.com/ep2.mp3"},
    ]
    fake_episodes = [_fake_episode("ep1.mp3"), _fake_episode("ep2.mp3")]
    fake_db = MagicMock()

    with patch("app.ingest.service.parse_feed", return_value=entries), \
         patch("app.ingest.service.ingest_from_url", side_effect=fake_episodes) as mock_ingest:
        result = ingest_from_rss("https://example.com/feed.xml", db=fake_db, limit=5)

    assert len(result["episodes"]) == 2
    assert result["errors"] == []
    assert mock_ingest.call_args_list[0].args == ("https://example.com/ep1.mp3", fake_db)
    assert mock_ingest.call_args_list[1].args == ("https://example.com/ep2.mp3", fake_db)


def test_ingest_from_rss_continues_after_one_failure():
    entries = [
        {"title": "Good", "audio_url": "https://example.com/good.mp3"},
        {"title": "Bad", "audio_url": "https://example.com/bad.mp3"},
    ]

    def side_effect(url, db):
        if "bad" in url:
            raise RuntimeError("download failed")
        return _fake_episode("good.mp3")

    with patch("app.ingest.service.parse_feed", return_value=entries), \
         patch("app.ingest.service.ingest_from_url", side_effect=side_effect):
        result = ingest_from_rss("https://example.com/feed.xml", db=MagicMock(), limit=5)

    assert len(result["episodes"]) == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["title"] == "Bad"
    assert "download failed" in result["errors"][0]["error"]


def test_ingest_from_rss_passes_limit_to_parse_feed():
    with patch("app.ingest.service.parse_feed", return_value=[]) as mock_parse, \
         patch("app.ingest.service.ingest_from_url"):
        ingest_from_rss("https://example.com/feed.xml", db=MagicMock(), limit=3)

    mock_parse.assert_called_once_with("https://example.com/feed.xml", limit=3)


def test_ingest_from_rss_empty_feed_returns_empty_result():
    with patch("app.ingest.service.parse_feed", return_value=[]), \
         patch("app.ingest.service.ingest_from_url") as mock_ingest:
        result = ingest_from_rss("https://example.com/feed.xml", db=MagicMock())

    assert result == {"episodes": [], "errors": []}
    mock_ingest.assert_not_called()


def test_ingest_from_rss_hides_unexpected_error_detail():
    entries = [{"title": "Bad", "audio_url": "https://example.com/bad.mp3"}]

    with patch("app.ingest.service.parse_feed", return_value=entries), \
         patch("app.ingest.service.ingest_from_url", side_effect=OSError("leaked /srv/secret/path")):
        result = ingest_from_rss("https://example.com/feed.xml", db=MagicMock())

    assert len(result["errors"]) == 1
    assert "leaked" not in result["errors"][0]["error"]
    assert "secret" not in result["errors"][0]["error"]
