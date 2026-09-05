import feedparser


def parse_feed(feed_url: str, limit: int = 5) -> list[dict]:
    """
    Parses a podcast RSS/Atom feed and returns up to `limit` episodes as
    [{"title": str, "audio_url": str}, ...] in feed order. Entries without
    an audio enclosure link are skipped (not counted against `limit`).
    """
    parsed = feedparser.parse(feed_url)
    episodes: list[dict] = []
    for entry in parsed.entries:
        audio_url = None
        for link in entry.get("links", []):
            if link.get("rel") == "enclosure" and link.get("type", "").startswith("audio"):
                audio_url = link.get("href")
                break
        if not audio_url:
            continue
        episodes.append({
            "title": entry.get("title") or "Untitled episode",
            "audio_url": audio_url,
        })
        if len(episodes) >= limit:
            break
    return episodes
