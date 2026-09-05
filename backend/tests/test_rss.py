from types import SimpleNamespace
from unittest.mock import patch

from app.ingest.rss import parse_feed


def _entry(title: str, audio_url: str | None, audio_type: str = "audio/mpeg") -> dict:
    links = []
    if audio_url:
        links.append({"rel": "enclosure", "type": audio_type, "href": audio_url})
    return {"title": title, "links": links}


def test_parse_feed_extracts_audio_enclosures():
    fake_feed = SimpleNamespace(entries=[
        _entry("Episode 1", "https://example.com/ep1.mp3"),
        _entry("Episode 2", "https://example.com/ep2.mp3"),
    ])
    with patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert result == [
        {"title": "Episode 1", "audio_url": "https://example.com/ep1.mp3"},
        {"title": "Episode 2", "audio_url": "https://example.com/ep2.mp3"},
    ]


def test_parse_feed_skips_entries_without_audio_enclosure():
    fake_feed = SimpleNamespace(entries=[
        _entry("Has audio", "https://example.com/ep1.mp3"),
        _entry("No audio", None),
    ])
    with patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert len(result) == 1
    assert result[0]["title"] == "Has audio"


def test_parse_feed_ignores_non_audio_enclosures():
    fake_feed = SimpleNamespace(entries=[
        _entry("Video episode", "https://example.com/ep1.mp4", audio_type="video/mp4"),
    ])
    with patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert result == []


def test_parse_feed_respects_limit():
    fake_feed = SimpleNamespace(entries=[
        _entry(f"Episode {i}", f"https://example.com/ep{i}.mp3") for i in range(10)
    ])
    with patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml", limit=3)

    assert len(result) == 3


def test_parse_feed_defaults_untitled_entry():
    fake_feed = SimpleNamespace(entries=[
        {"links": [{"rel": "enclosure", "type": "audio/mpeg", "href": "https://example.com/ep1.mp3"}]},
    ])
    with patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert result[0]["title"] == "Untitled episode"
