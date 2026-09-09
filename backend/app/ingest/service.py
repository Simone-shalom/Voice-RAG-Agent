import logging
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from .chunker import chunk_segments
from .diarizer import assign_speaker_labels, diarize_audio
from .embedder import embed_texts
from .rss import parse_feed
from .summarizer import summarize_episode
from .transcriber import transcribe
from ..core.url_safety import assert_safe_url
from ..db.models import Chunk, Episode

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB cap on ingest_from_url downloads


def ingest_audio(audio_path: Path, filename: str, source_url: str | None, db: Session) -> Episode:
    segments = transcribe(audio_path)
    chunk_dicts = chunk_segments(segments)
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)

    speaker_labels: list[str | None] = [None] * len(chunk_dicts)
    try:
        words = diarize_audio(audio_path)
        speaker_labels = assign_speaker_labels(chunk_dicts, words)
    except Exception:
        # Diarization is a nice-to-have, never a reason to fail an ingest —
        # the episode must stay fully searchable even if Deepgram is down.
        logger.exception("diarization failed — continuing without speaker labels")

    # Merge the diarizer's generic "Speaker N" labels onto the chunks BEFORE
    # summarizing: summarize_episode's speaker_names mapping is only grounded
    # in reality if Claude can actually see which chunk belongs to which
    # speaker index. Without this, any "Speaker N" -> real-name mapping it
    # returns is a guess about an index it was never shown, and applying that
    # guess to Chunk.speaker_label would mislabel real speech with the wrong
    # person's name — worse than the honest generic label it replaces. If
    # diarization failed above, every label here is just None, which
    # summarize_episode treats as "unlabeled" (no prefix), not an error.
    for chunk_dict, label in zip(chunk_dicts, speaker_labels):
        chunk_dict["speaker_label"] = label

    summary: str | None = None
    chapters: list[dict] | None = None
    try:
        summary_result = summarize_episode(chunk_dicts)
        summary = summary_result["summary"] or None
        chapters = summary_result["chapters"] or None
        speaker_names = summary_result["speaker_names"]
        if speaker_names:
            speaker_labels = [speaker_names.get(label, label) if label else None for label in speaker_labels]
    except Exception:
        # Same rationale as diarization above — summarization must never
        # abort an otherwise-successful ingest.
        logger.exception("summarization failed — continuing without summary/chapters")

    episode = Episode(id=uuid.uuid4(), filename=filename, source_url=source_url, summary=summary, chapters=chapters)
    db.add(episode)

    for chunk_dict, embedding, speaker_label in zip(chunk_dicts, embeddings, speaker_labels):
        db.add(
            Chunk(
                episode_id=episode.id,
                start_ts=chunk_dict["start_ts"],
                end_ts=chunk_dict["end_ts"],
                text=chunk_dict["text"],
                embedding=embedding,
                speaker_label=speaker_label,
            )
        )

    db.commit()
    db.refresh(episode)
    return episode


def _validate_redirect(request: httpx.Request) -> None:
    """httpx 'request' event hook — re-validates every hop of a redirect chain."""
    assert_safe_url(str(request.url))


def ingest_from_url(url: str, db: Session) -> Episode:
    assert_safe_url(url)
    url_path = Path(urlparse(url).path)
    suffix = url_path.suffix or ".mp3"
    filename = url_path.name or "audio"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = Path(f.name)

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=60.0,
            event_hooks={"request": [_validate_redirect]},
        ) as http:
            with http.stream("GET", url) as response:
                response.raise_for_status()
                total = 0
                with open(tmp_path, "wb") as out:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} byte limit")
                        out.write(chunk)
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
        except (ValueError, RuntimeError) as e:
            errors.append({"title": entry["title"], "error": str(e)})
        except Exception:
            logger.exception("RSS batch episode ingest failed: %s", entry["title"])
            errors.append({"title": entry["title"], "error": "internal error — see server logs"})

    return {"episodes": episodes, "errors": errors}
