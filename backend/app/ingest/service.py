import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from .chunker import chunk_segments
from .embedder import embed_texts
from .rss import parse_feed
from .transcriber import transcribe
from ..db.models import Chunk, Episode


def ingest_audio(audio_path: Path, filename: str, source_url: str | None, db: Session) -> Episode:
    segments = transcribe(audio_path)
    chunk_dicts = chunk_segments(segments)
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)

    episode = Episode(id=uuid.uuid4(), filename=filename, source_url=source_url)
    db.add(episode)

    for chunk_dict, embedding in zip(chunk_dicts, embeddings):
        db.add(
            Chunk(
                episode_id=episode.id,
                start_ts=chunk_dict["start_ts"],
                end_ts=chunk_dict["end_ts"],
                text=chunk_dict["text"],
                embedding=embedding,
            )
        )

    db.commit()
    db.refresh(episode)
    return episode


def ingest_from_url(url: str, db: Session) -> Episode:
    url_path = Path(urlparse(url).path)
    suffix = url_path.suffix or ".mp3"
    filename = url_path.name or "audio"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = Path(f.name)
        with httpx.Client(follow_redirects=True, timeout=60.0) as http:
            with http.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    f.write(chunk)

    try:
        return ingest_audio(tmp_path, filename, url, db)
    finally:
        tmp_path.unlink(missing_ok=True)


def ingest_from_rss(feed_url: str, db: Session, limit: int = 5) -> dict:
    """
    Parses `feed_url` and ingests up to `limit` episodes via ingest_from_url.
    A failure on one episode is captured, not raised — the batch continues.

    Returns {"episodes": [Episode, ...], "errors": [{"title": str, "error": str}, ...]}.
    """
    entries = parse_feed(feed_url, limit=limit)
    episodes: list[Episode] = []
    errors: list[dict] = []

    for entry in entries:
        try:
            episode = ingest_from_url(entry["audio_url"], db)
            episodes.append(episode)
        except Exception as e:
            errors.append({"title": entry["title"], "error": str(e)})

    return {"episodes": episodes, "errors": errors}
