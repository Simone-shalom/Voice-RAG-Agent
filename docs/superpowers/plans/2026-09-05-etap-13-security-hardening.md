# Etap 13 — Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the critical/high-severity security gaps found in the pre-Etap-13 audit (SSRF via URL/RSS ingest, unbounded uploads, unvalidated file content, verbose error disclosure, no rate limiting, no CORS policy) before any public deployment (Etap 11).

**Architecture:** Add a small `backend/app/core/` package for cross-cutting security utilities (URL safety, auth, rate limiting, logging config) that existing routers/services import and call at their existing entry points. No new services, no new DB tables, no change to the public JSON contract of any endpoint (only internal parameter names and error-response bodies for the *unexpected*-error path change).

**Tech Stack:** Python 3.12, FastAPI, httpx, `slowapi` (new dependency) for rate limiting, stdlib `socket`/`ipaddress`/`logging`.

**Spec:** This plan is derived directly from a multi-agent security/deployment audit run against `backend/app/` and `frontend/` on 2026-09-05 (no separate spec doc — findings are summarized inline per task).

## Global Constraints

- Every existing backend test must keep passing except the two tests explicitly called out for a deliberate behavior change (Task 3). Run `cd backend && python -m pytest tests/ -v` after every task.
- No test may require real network access, a real database, or real API keys — mock `socket.getaddrinfo`, `httpx.Client`, and external SDKs exactly like the existing suite does (see `backend/tests/conftest.py` and any existing test file in the same module for the established mocking pattern).
- Do NOT commit after individual tasks. Tasks run sequentially in the current working tree; someone else makes ONE combined commit at the end of the etap (per project convention in `CLAUDE.md` — "One squash-commit per etap on `main`"). Do not run `git commit` in any task.
- No `Co-Authored-By` / attribution lines anywhere (commit message is written at the end of the etap by the coordinator, not by task implementers).
- Follow existing code conventions: routers raise `HTTPException`, services raise `ValueError`/`RuntimeError` for expected failures, tests use `unittest.mock.patch` against the *importing* module's namespace (e.g. `patch("app.ingest.service.httpx.Client", ...)`, not `patch("httpx.Client", ...)`).

---

### Task 1: SSRF guard for outbound URL fetches (RSS + URL ingest)

**Finding this fixes:** `ingest_from_rss`/`parse_feed` call `feedparser.parse(feed_url)` and `ingest_from_url` calls `httpx` on a fully user-controlled URL with no scheme or destination-address check. An attacker can submit `file:///etc/passwd`, `http://169.254.169.254/...` (cloud metadata), or `http://localhost:5433` (internal service) via `POST /ingest/url` or `POST /ingest/rss` and have the server fetch it on their behalf.

**Files:**
- Create: `backend/app/core/__init__.py` (empty)
- Create: `backend/app/core/url_safety.py`
- Test: `backend/tests/core/__init__.py` (empty)
- Test: `backend/tests/core/test_url_safety.py`
- Modify: `backend/app/ingest/rss.py`
- Modify: `backend/app/ingest/service.py`
- Modify (tests): `backend/tests/test_rss.py`
- Create (tests): `backend/tests/test_ingest_url.py`

**Interfaces:**
- Produces: `app.core.url_safety.assert_safe_url(url: str) -> None` — raises `ValueError` if `url` is unsafe to fetch (non-http(s) scheme, or hostname resolves to a private/loopback/link-local/reserved/multicast address). Later tasks do not depend on this, but Task 4/5 leave it untouched.

- [ ] **Step 1: Write the failing tests for `assert_safe_url`**

Create `backend/tests/core/__init__.py` (empty file) and `backend/tests/core/test_url_safety.py`:

```python
import socket
from unittest.mock import patch

import pytest

from app.core.url_safety import assert_safe_url


def _addrinfo(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme"):
        assert_safe_url("file:///etc/passwd")


def test_rejects_url_with_no_hostname():
    with pytest.raises(ValueError, match="hostname"):
        assert_safe_url("http://")


def test_rejects_unresolvable_host():
    with patch("app.core.url_safety.socket.getaddrinfo", side_effect=socket.gaierror("nope")):
        with pytest.raises(ValueError, match="could not resolve"):
            assert_safe_url("http://nonexistent.example")


def test_rejects_loopback_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://localhost/feed.xml")


def test_rejects_private_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("10.0.0.5")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://internal.example/feed.xml")


def test_rejects_link_local_metadata_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://metadata.example/latest")


def test_allows_public_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
        assert_safe_url("https://example.com/feed.xml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/core/test_url_safety.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.core'`)

- [ ] **Step 3: Implement `assert_safe_url`**

Create `backend/app/core/__init__.py` — empty file.

Create `backend/app/core/url_safety.py`:

```python
import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}


def assert_safe_url(url: str) -> None:
    """
    Raises ValueError if `url` is unsafe for the server to fetch: a
    non-http(s) scheme, or a hostname that resolves to a private,
    loopback, link-local, reserved, or multicast address. Blocks SSRF
    against internal services and cloud metadata endpoints reachable
    from server-side URL/RSS ingest.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host: {parsed.hostname}") from e

    for _family, _type, _proto, _canon, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"URL resolves to a disallowed address: {ip}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/core/test_url_safety.py -v`
Expected: 7 passed

- [ ] **Step 5: Wire `assert_safe_url` into `parse_feed` and update its tests**

Modify `backend/app/ingest/rss.py` (add the import and the guard call as the first line of the function body):

```python
import feedparser

from ..core.url_safety import assert_safe_url


def parse_feed(feed_url: str, limit: int = 5) -> list[dict]:
    """
    Parses a podcast RSS/Atom feed and returns up to `limit` episodes as
    [{"title": str, "audio_url": str}, ...] in feed order. Entries without
    an audio enclosure link are skipped (not counted against `limit`).
    """
    assert_safe_url(feed_url)
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
```

The 5 existing tests in `backend/tests/test_rss.py` call the real `parse_feed()` and only mock `feedparser.parse` — without also mocking `assert_safe_url`, they would now trigger a real DNS lookup for `example.com`. Rewrite the whole file to mock both:

```python
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
    with patch("app.ingest.rss.assert_safe_url"), \
         patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
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
    with patch("app.ingest.rss.assert_safe_url"), \
         patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert len(result) == 1
    assert result[0]["title"] == "Has audio"


def test_parse_feed_ignores_non_audio_enclosures():
    fake_feed = SimpleNamespace(entries=[
        _entry("Video episode", "https://example.com/ep1.mp4", audio_type="video/mp4"),
    ])
    with patch("app.ingest.rss.assert_safe_url"), \
         patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert result == []


def test_parse_feed_respects_limit():
    fake_feed = SimpleNamespace(entries=[
        _entry(f"Episode {i}", f"https://example.com/ep{i}.mp3") for i in range(10)
    ])
    with patch("app.ingest.rss.assert_safe_url"), \
         patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml", limit=3)

    assert len(result) == 3


def test_parse_feed_defaults_untitled_entry():
    fake_feed = SimpleNamespace(entries=[
        {"links": [{"rel": "enclosure", "type": "audio/mpeg", "href": "https://example.com/ep1.mp3"}]},
    ])
    with patch("app.ingest.rss.assert_safe_url"), \
         patch("app.ingest.rss.feedparser.parse", return_value=fake_feed):
        result = parse_feed("https://example.com/feed.xml")

    assert result[0]["title"] == "Untitled episode"


def test_parse_feed_rejects_unsafe_url():
    with pytest.raises(ValueError):
        parse_feed("file:///etc/passwd")
```

Add `import pytest` to the top of the file alongside the existing imports for the new last test.

- [ ] **Step 6: Wire `assert_safe_url` + a redirect-safe client + a download-size cap into `ingest_from_url`**

Modify `backend/app/ingest/service.py`:

```python
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
from ..core.url_safety import assert_safe_url
from ..db.models import Chunk, Episode

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB cap on ingest_from_url downloads


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
        except Exception as e:
            errors.append({"title": entry["title"], "error": str(e)})

    return {"episodes": episodes, "errors": errors}
```

Note the cleanup fix bundled in: `tmp_path.unlink()` now runs in a `finally` around the *entire* download+ingest, not just around `ingest_audio` — previously an HTTP error or (new) size-cap error during download would leak the temp file.

- [ ] **Step 7: Write new tests for `ingest_from_url` (previously untested)**

Create `backend/tests/test_ingest_url.py`:

```python
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
```

- [ ] **Step 8: Run the full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass (159 previous + ~11 new from this task).

---

### Task 2: Upload size cap + audio content-type sniffing + RSS batch-size cap

**Finding this fixes:** `/ingest/upload` reads the entire file into memory with no size limit and no check that the content is actually audio (any binary blob is handed to Whisper). `POST /ingest/rss` accepts an unbounded `limit` (client-side cap of 20 in the frontend is not enforced server-side).

**Files:**
- Create: `backend/app/ingest/validators.py`
- Test: `backend/tests/test_ingest_validators.py`
- Modify: `backend/app/ingest/router.py`
- Modify (tests): `backend/tests/test_ingest_router.py`

**Interfaces:**
- Produces: `app.ingest.validators.MAX_UPLOAD_BYTES: int`, `read_capped(fileobj, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes`, `is_probably_audio(header: bytes) -> bool`.

- [ ] **Step 1: Write the failing tests for the validators module**

Create `backend/tests/test_ingest_validators.py`:

```python
import io

import pytest

from app.ingest.validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped


def test_recognizes_mp3_with_id3_tag():
    assert is_probably_audio(b"ID3\x03\x00\x00\x00")


def test_recognizes_mp3_frame_sync_without_id3():
    assert is_probably_audio(b"\xff\xfb\x90\x00")


def test_recognizes_wav():
    assert is_probably_audio(b"RIFF\x00\x00\x00\x00WAVEfmt ")


def test_recognizes_ogg():
    assert is_probably_audio(b"OggS\x00\x02\x00\x00")


def test_recognizes_m4a():
    assert is_probably_audio(b"\x00\x00\x00\x18ftypM4A ")


def test_recognizes_webm():
    assert is_probably_audio(b"\x1a\x45\xdf\xa3\x01\x00\x00\x00")


def test_rejects_unknown_binary():
    assert not is_probably_audio(b"PK\x03\x04random zip bytes")


def test_rejects_empty_header():
    assert not is_probably_audio(b"")


def test_read_capped_returns_full_content_under_limit():
    data = b"a" * 100
    assert read_capped(io.BytesIO(data), max_bytes=1000) == data


def test_read_capped_raises_when_over_limit():
    data = b"a" * 2000
    with pytest.raises(ValueError, match="exceeds"):
        read_capped(io.BytesIO(data), max_bytes=1000)


def test_max_upload_bytes_is_reasonable():
    assert 1_000_000 < MAX_UPLOAD_BYTES <= 500 * 1024 * 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_ingest_validators.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.ingest.validators'`)

- [ ] **Step 3: Implement the validators module**

Create `backend/app/ingest/validators.py`:

```python
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

_MAGIC_SIGNATURES = (
    (0, b"ID3"),               # MP3 with ID3 tag
    (0, b"OggS"),              # Ogg
    (0, b"RIFF"),              # WAV (RIFF....WAVE container)
    (4, b"ftyp"),              # MP4 / M4A
    (0, b"\x1a\x45\xdf\xa3"),  # WebM / Matroska (EBML header)
)
_MPEG_FRAME_SYNCS = (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"\xff\xf2")


def is_probably_audio(header: bytes) -> bool:
    """
    Sniffs the first bytes of a file for known audio-container magic
    numbers. Conservative allow-list (not exhaustive — e.g. a bare MPEG
    frame with an unusual sync byte still passes since Whisper handles it)
    — rejects obviously non-audio uploads without a libmagic dependency.
    """
    if header[:2] in _MPEG_FRAME_SYNCS:
        return True
    return any(header[offset:offset + len(sig)] == sig for offset, sig in _MAGIC_SIGNATURES)


def read_capped(fileobj, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Reads fileobj in chunks, raising ValueError if it exceeds max_bytes."""
    chunks = []
    total = 0
    while True:
        chunk = fileobj.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"upload exceeds {max_bytes} byte limit")
        chunks.append(chunk)
    return b"".join(chunks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_ingest_validators.py -v`
Expected: 10 passed

- [ ] **Step 5: Wire the cap + sniff into `/ingest/upload`, and cap `limit` on `/ingest/rss`**

Modify `backend/app/ingest/router.py`:

```python
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db.session import get_db
from .service import ingest_audio, ingest_from_rss, ingest_from_url
from .validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int


class IngestRSSRequest(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=20)


class RSSErrorItem(BaseModel):
    title: str
    error: str


class RSSIngestResponse(BaseModel):
    episodes: list[EpisodeResponse]
    errors: list[RSSErrorItem]


@router.post("/upload", response_model=EpisodeResponse)
def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        data = read_capped(file.file, MAX_UPLOAD_BYTES)
        if not is_probably_audio(data[:16]):
            raise ValueError("file does not look like a supported audio format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/url", response_model=EpisodeResponse)
def ingest_url(body: IngestURLRequest, db: Session = Depends(get_db)):
    try:
        episode = ingest_from_url(body.url, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/rss", response_model=RSSIngestResponse)
def ingest_rss(body: IngestRSSRequest, db: Session = Depends(get_db)):
    try:
        result = ingest_from_rss(body.url, db, limit=body.limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RSSIngestResponse(
        episodes=[
            EpisodeResponse(id=str(ep.id), filename=ep.filename, chunk_count=len(ep.chunks))
            for ep in result["episodes"]
        ],
        errors=[RSSErrorItem(**err) for err in result["errors"]],
    )
```

(The `except Exception` blocks stay broad for now — Task 3 narrows them.)

- [ ] **Step 6: Fix existing upload tests that send non-audio bytes, and add new coverage**

The existing `backend/tests/test_ingest_router.py` sends `BytesIO(b"fake audio")` in three tests, which the new content sniff now rejects with a 400 before ever reaching the mocked `ingest_audio` — breaking their intent. Replace the payload with an ID3-prefixed one so the sniff passes and the test still exercises the mocked behavior it's named for. Apply this exact substitution three times: change `BytesIO(b"fake audio")` to `BytesIO(b"ID3fake audio")` in:
- `test_upload_returns_episode_id`
- `test_upload_returns_400_on_ingest_failure`
- `test_upload_returns_400_on_tempfile_write_failure`

Then append these new tests to the end of `backend/tests/test_ingest_router.py`:

```python
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
```

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass.

---

### Task 3: Generic error responses for unexpected failures + server-side logging

**Finding this fixes:** `except Exception as e: raise HTTPException(status_code=400, detail=str(e))` in the ingest, search, and agent-chat routers echoes raw exception text (which can include upstream API error bodies, stack trace fragments, or filesystem paths) straight back to the caller, with nothing logged server-side to debug the failure after the fact.

**Files:**
- Create: `backend/app/core/logging.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/ingest/router.py`
- Modify: `backend/app/search/router.py`
- Modify: `backend/app/agent/router.py`
- Modify (tests): `backend/tests/test_ingest_router.py`, `backend/tests/test_search_router.py`, `backend/tests/test_agent_router.py`

**Interfaces:**
- Produces: `app.core.logging.configure_logging() -> None`.
- Behavior change: a router now returns 400 with the real message ONLY when the underlying code raised `ValueError`/`RuntimeError` (our own, deliberately-worded exceptions). Any other exception type now returns 500 with a fixed generic message, and the real exception is logged server-side via `logger.exception(...)`.

- [ ] **Step 1: Update the one existing test whose expected behavior changes, and add new tests for the generic-500 path**

`test_upload_returns_400_on_tempfile_write_failure` in `backend/tests/test_ingest_router.py` currently raises `OSError("disk full")` and expects a 400 with that text — `OSError` is not one of our deliberately-raised types, so it now falls into the generic-500 branch. Replace that test:

```python
def test_upload_returns_500_and_hides_detail_on_unexpected_failure(client):
    with patch("app.ingest.router.tempfile.NamedTemporaryFile", side_effect=OSError("disk full at /var/lib/secret")):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 500
    assert "disk full" not in response.json()["detail"]
    assert "secret" not in response.json()["detail"]
```

Append equivalent generic-failure tests for the other two ingest endpoints, search, and agent chat:

```python
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
```

Append to `backend/tests/test_search_router.py`:

```python
def test_search_returns_500_and_hides_detail_on_unexpected_failure(client):
    with patch("app.search.router.semantic_search", side_effect=OSError("leaked connection string")):
        response = client.get("/search", params={"q": "test"})

    assert response.status_code == 500
    assert "leaked connection string" not in response.json()["detail"]
```

Append to `backend/tests/test_agent_router.py`:

```python
def test_chat_returns_500_and_hides_detail_on_unexpected_failure(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("app.agent.router.build_graph", side_effect=OSError("leaked internal path")):
        response = client.post("/agent/chat", json={"message": "hi", "thread_id": "t1"})

    assert response.status_code == 500
    assert "leaked internal path" not in response.json()["detail"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_ingest_router.py tests/test_search_router.py tests/test_agent_router.py -v -k "unexpected_failure or tempfile_write_failure"`
Expected: FAIL — all still return 400 with the leaked text, since routers aren't updated yet.

- [ ] **Step 3: Add the logging config module**

Create `backend/app/core/logging.py`:

```python
import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
```

- [ ] **Step 4: Call it at app startup**

Modify `backend/app/main.py` — add the import and call it before the `FastAPI(...)` construction:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.logging import configure_logging
from .db.session import engine, init_db
from .agent.router import router as agent_router
from .episodes.router import router as episodes_router
from .ingest.router import router as ingest_router
from .search.router import router as search_router
from .voice.router import router as voice_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(episodes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 5: Narrow the except blocks in the ingest router**

Modify `backend/app/ingest/router.py` — add `import logging` and `logger = logging.getLogger(__name__)` near the top, then replace each of the three `except Exception as e: raise HTTPException(status_code=400, detail=str(e))` blocks:

```python
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db.session import get_db
from .service import ingest_audio, ingest_from_rss, ingest_from_url
from .validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int


class IngestRSSRequest(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=20)


class RSSErrorItem(BaseModel):
    title: str
    error: str


class RSSIngestResponse(BaseModel):
    episodes: list[EpisodeResponse]
    errors: list[RSSErrorItem]


@router.post("/upload", response_model=EpisodeResponse)
def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        data = read_capped(file.file, MAX_UPLOAD_BYTES)
        if not is_probably_audio(data[:16]):
            raise ValueError("file does not look like a supported audio format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest upload failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/url", response_model=EpisodeResponse)
def ingest_url(body: IngestURLRequest, db: Session = Depends(get_db)):
    try:
        episode = ingest_from_url(body.url, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest from URL failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/rss", response_model=RSSIngestResponse)
def ingest_rss(body: IngestRSSRequest, db: Session = Depends(get_db)):
    try:
        result = ingest_from_rss(body.url, db, limit=body.limit)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest from RSS failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return RSSIngestResponse(
        episodes=[
            EpisodeResponse(id=str(ep.id), filename=ep.filename, chunk_count=len(ep.chunks))
            for ep in result["episodes"]
        ],
        errors=[RSSErrorItem(**err) for err in result["errors"]],
    )
```

- [ ] **Step 6: Narrow the except block in the search router**

Modify `backend/app/search/router.py`:

```python
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .bm25 import bm25_search
from .searcher import hybrid_rerank_search, hybrid_search, semantic_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

SearchMode = Literal["semantic", "bm25", "hybrid", "hybrid+rerank"]


class SearchResult(BaseModel):
    chunk_id: str
    episode_id: str
    start_ts: float
    end_ts: float
    text: str
    similarity: float


@router.get("", response_model=list[SearchResult])
def search(
    q: str = Query(..., min_length=1),
    mode: SearchMode = Query("semantic"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    try:
        if mode == "semantic":
            return semantic_search(query=q, limit=limit, db=db)
        elif mode == "bm25":
            return bm25_search(query=q, limit=limit, db=db)
        elif mode == "hybrid":
            return hybrid_search(query=q, limit=limit, db=db)
        else:  # hybrid+rerank
            return hybrid_rerank_search(query=q, limit=limit, db=db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
```

- [ ] **Step 7: Narrow the except block in the agent chat endpoint**

Modify `backend/app/agent/router.py` — only the `chat` function's except block changes (leave `chat_stream` as-is; its errors go through `stream_agent_response`'s own SSE `{"type": "error"}` event, which is a different, already-reviewed error path):

```python
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .graph import build_graph
from .streaming import stream_agent_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    reply: str
    steps: int


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)], "step_count": 0},
            config={"configurable": {"thread_id": request.thread_id}},
        )
        last_msg = result["messages"][-1]
        return ChatResponse(reply=last_msg.content, steps=result["step_count"])
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("agent chat failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=request.message,
            thread_id=request.thread_id,
            db=db,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 8: Run the full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass, including the updated/new ones from Step 1.

---

### Task 4: Rate limiting on cost-bearing endpoints + optional API-key gate on ingest

**Finding this fixes:** No endpoint has any rate limit, and every endpoint that calls a paid external API (`/agent/chat`, `/agent/chat/stream`, `/voice/stt`, `/voice/tts`, `/ingest/*`) is reachable by anyone with no cost control. Ingest endpoints in particular can trigger arbitrarily many paid Whisper transcriptions.

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/core/rate_limit.py`
- Create: `backend/app/core/auth.py`
- Test: `backend/tests/core/test_auth.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/ingest/router.py`
- Modify: `backend/app/voice/router.py`
- Modify: `backend/app/agent/router.py`
- Modify: `backend/tests/conftest.py`
- Modify (tests): `backend/tests/test_ingest_router.py`

**Interfaces:**
- Produces: `app.core.rate_limit.limiter` (a `slowapi.Limiter` instance, `enabled` reflects the `RATE_LIMIT_ENABLED` env var, default disabled so the existing suite is unaffected). `app.core.auth.require_api_key` (a FastAPI dependency; no-op unless the `API_KEY` env var is set).

- [ ] **Step 1: Add the dependency and the disabled-by-default test env var**

Append to `backend/requirements.txt`:

```
slowapi==0.1.9
```

Modify `backend/tests/conftest.py` — add one more `setdefault` so the whole suite runs with rate limiting off unless a specific test opts in:

```python
import os

os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# init_db is now in the lifespan context; TestClient without `with` does not trigger it.
from app.main import app
from app.db.session import get_db


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Write the failing test for the API-key dependency**

Create `backend/tests/core/test_auth.py`:

```python
import pytest
from fastapi import HTTPException

from app.core.auth import require_api_key


def test_allows_request_when_api_key_env_unset(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    require_api_key(x_api_key=None)  # does not raise


def test_rejects_missing_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_rejects_wrong_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    with pytest.raises(HTTPException):
        require_api_key(x_api_key="wrong")


def test_allows_correct_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    require_api_key(x_api_key="secret123")  # does not raise
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd backend && python -m pytest tests/core/test_auth.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.core.auth'`)

- [ ] **Step 4: Implement the auth dependency and the rate limiter**

Create `backend/app/core/auth.py`:

```python
import os

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Gates a router behind a shared API key passed as the X-API-Key header.
    No-op when the API_KEY env var is unset (local dev / test default) —
    only enforced once an operator opts in for a public deployment.
    """
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
```

Create `backend/app/core/rate_limit.py`:

```python
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    enabled=os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true",
)
```

- [ ] **Step 5: Run the auth test to verify it passes**

Run: `cd backend && python -m pytest tests/core/test_auth.py -v`
Expected: 4 passed

- [ ] **Step 6: Wire the limiter into the app**

Modify `backend/app/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from .core.logging import configure_logging
from .core.rate_limit import limiter
from .db.session import engine, init_db
from .agent.router import router as agent_router
from .episodes.router import router as episodes_router
from .ingest.router import router as ingest_router
from .search.router import router as search_router
from .voice.router import router as voice_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(episodes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 7: Apply rate limits + the API-key gate to the ingest router**

Modify `backend/app/ingest/router.py` — add the imports, apply `dependencies=[Depends(require_api_key)]` at the router level (all three ingest endpoints are write/cost operations a demo visitor doesn't need), add a `request: Request` parameter and a `@limiter.limit(...)` decorator to each endpoint:

```python
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.auth import require_api_key
from ..core.rate_limit import limiter
from ..db.session import get_db
from .service import ingest_audio, ingest_from_rss, ingest_from_url
from .validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int


class IngestRSSRequest(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=20)


class RSSErrorItem(BaseModel):
    title: str
    error: str


class RSSIngestResponse(BaseModel):
    episodes: list[EpisodeResponse]
    errors: list[RSSErrorItem]


@router.post("/upload", response_model=EpisodeResponse)
@limiter.limit("10/minute")
def upload_audio(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        data = read_capped(file.file, MAX_UPLOAD_BYTES)
        if not is_probably_audio(data[:16]):
            raise ValueError("file does not look like a supported audio format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest upload failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/url", response_model=EpisodeResponse)
@limiter.limit("10/minute")
def ingest_url(request: Request, body: IngestURLRequest, db: Session = Depends(get_db)):
    try:
        episode = ingest_from_url(body.url, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest from URL failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/rss", response_model=RSSIngestResponse)
@limiter.limit("5/minute")
def ingest_rss(request: Request, body: IngestRSSRequest, db: Session = Depends(get_db)):
    try:
        result = ingest_from_rss(body.url, db, limit=body.limit)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest from RSS failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return RSSIngestResponse(
        episodes=[
            EpisodeResponse(id=str(ep.id), filename=ep.filename, chunk_count=len(ep.chunks))
            for ep in result["episodes"]
        ],
        errors=[RSSErrorItem(**err) for err in result["errors"]],
    )
```

- [ ] **Step 8: Apply rate limits to the voice router**

Modify `backend/app/voice/router.py` — rename the TTS body parameter (it collides with slowapi's required `request: Request` name) and add limits:

```python
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..core.rate_limit import limiter
from .stt import transcribe_audio
from .tts import synthesise

router = APIRouter(prefix="/voice", tags=["voice"])


class STTResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


@router.post("/stt", response_model=STTResponse)
@limiter.limit("20/minute")
async def speech_to_text(request: Request, audio: UploadFile = File(...)):
    try:
        file_bytes = await audio.read()
        text = transcribe_audio(file_bytes, audio.filename or "audio.webm")
        return STTResponse(text=text)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tts")
@limiter.limit("20/minute")
def text_to_speech(request: Request, body: TTSRequest):
    try:
        audio_bytes = synthesise(body.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 9: Apply rate limits to the agent router**

Modify `backend/app/agent/router.py` — rename both bodies to `body` (collision with `request: Request`) and add limits:

```python
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.rate_limit import limiter
from ..db.session import get_db
from .graph import build_graph
from .streaming import stream_agent_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    reply: str
    steps: int


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=body.message)], "step_count": 0},
            config={"configurable": {"thread_id": body.thread_id}},
        )
        last_msg = result["messages"][-1]
        return ChatResponse(reply=last_msg.content, steps=result["step_count"])
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("agent chat failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=body.message,
            thread_id=body.thread_id,
            db=db,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 10: Add tests proving the API-key gate and the rate limiter actually work end-to-end**

Append to `backend/tests/test_ingest_router.py`:

```python
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
```

- [ ] **Step 11: Run the full backend suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass (rate limiting stays disabled for every test except the one that explicitly flips `limiter.enabled`).

---

### Task 5: CORS policy, env-driven frontend backend URL, TTS text cap, Next.js patch bump

**Finding this fixes:** No CORS policy exists (blocks a split-domain Vercel+Railway deploy), `frontend/next.config.ts` hardcodes `http://localhost:8000` (breaks in any deployed environment), TTS accepts unbounded text length (unbounded ElevenLabs cost per call), and `next` is pinned to `15.0.0` (predates CVE-2025-29927 and later 15.x fixes).

**Files:**
- Modify: `backend/app/main.py`
- Modify: `.env.example`
- Modify: `frontend/next.config.ts`
- Modify: `frontend/package.json`
- Modify: `backend/app/voice/tts.py`
- Modify (tests): `backend/tests/test_voice_tts.py`

- [ ] **Step 1: Write the failing test for the TTS length cap**

Read `backend/tests/test_voice_tts.py` first to match its existing mocking style, then append:

```python
def test_synthesise_rejects_text_over_length_cap():
    from app.voice.tts import MAX_TTS_CHARS

    with pytest.raises(ValueError, match="exceeds"):
        synthesise("a" * (MAX_TTS_CHARS + 1))
```

(Add `import pytest` at the top of the file if not already present, and check the existing import line for `synthesise` to match — likely `from app.voice.tts import synthesise`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && python -m pytest tests/test_voice_tts.py -v -k length_cap`
Expected: FAIL (`ImportError: cannot import name 'MAX_TTS_CHARS'`)

- [ ] **Step 3: Add the cap to `synthesise`**

Modify `backend/app/voice/tts.py`:

```python
import os

import httpx

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George"
MAX_TTS_CHARS = 4000


def synthesise(text: str, voice_id: str = DEFAULT_VOICE_ID) -> bytes:
    """
    Convert text to MP3 audio using ElevenLabs TTS REST API (via httpx).
    Returns raw MP3 bytes.
    Raises ValueError for empty/oversized text, RuntimeError if API key unset.
    """
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"text exceeds {MAX_TTS_CHARS} character limit")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set — cannot use TTS")

    response = httpx.post(
        ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content
```

- [ ] **Step 4: Run it to verify it passes, then run the full backend suite**

Run: `cd backend && python -m pytest tests/test_voice_tts.py -v`
Expected: all pass.

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Add CORS middleware, driven by an env var**

Modify `backend/app/main.py` — add the import, read `ALLOWED_ORIGINS`, and register the middleware after `app = FastAPI(...)`:

```python
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from .core.logging import configure_logging
from .core.rate_limit import limiter
from .db.session import engine, init_db
from .agent.router import router as agent_router
from .episodes.router import router as episodes_router
from .ingest.router import router as ingest_router
from .search.router import router as search_router
from .voice.router import router as voice_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Voice Knowledge Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(episodes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

Append to `.env.example` (keep the existing Polish comments as-is for the rest of the file — this is a new section):

```
# CORS - dozwolone originy frontendu, oddzielone przecinkami (Etap 13)
ALLOWED_ORIGINS=http://localhost:3000

# Rate limiting (Etap 13) - "false" wylacza limity (np. do lokalnych testow)
RATE_LIMIT_ENABLED=true

# Opcjonalny klucz API wymagany w naglowku X-API-Key dla /ingest/* (Etap 13).
# Pozostaw puste, aby nie wymagac klucza (tryb demo/local dev).
API_KEY=
```

- [ ] **Step 6: Run the full backend suite once more**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass (CORS middleware doesn't affect same-origin `TestClient` calls).

- [ ] **Step 7: Make the frontend's backend URL env-driven**

Modify `frontend/next.config.ts`:

```typescript
import type { NextConfig } from "next";

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
```

(`docker-compose.yml` already sets `NEXT_PUBLIC_API_URL: http://localhost:8000` for the `frontend` service, so local behavior is unchanged; a production deploy sets it to the real backend URL instead.)

- [ ] **Step 8: Bump Next.js to a patched 15.x release**

Modify `frontend/package.json` — change line 12 from `"next": "15.0.0"` to `"next": "15.5.4"`.

- [ ] **Step 9: Reinstall and verify the frontend still builds and type-checks**

Run: `docker compose restart frontend` (Windows/Docker doesn't pick up `package.json` changes via hot-reload — see project gotcha)
Run: `docker compose exec frontend npm install`
Run: `docker compose exec frontend sh -c "npx tsc --noEmit"`
Expected: no type errors, container comes back up healthy.

- [ ] **Step 10: Verify the full Docker stack is healthy**

Run: `docker compose ps`
Expected: `postgres`, `backend`, `frontend` all `Up`.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers SSRF (audit findings #1, #2). Task 2 covers unbounded upload/download and missing content validation (#5, #6) and the unbounded RSS `limit` (#5). Task 3 covers verbose error disclosure (#7) and the missing-logging deployment finding. Task 4 covers no rate limiting (#4) and no auth (#3) — scoped to ingest only, deliberately leaving chat/search/voice open behind rate limits so the public demo stays usable without a login. Task 5 covers the CORS gap (#8) together with its tightly-coupled deployment twin (hardcoded `next.config.ts` URL), the TTS cost cap (#12), and the outdated Next.js pin (#10). Findings #9 (unauthenticated transcript reads) and #11 (redirect-hardening, folded into Task 1's `event_hooks` fix) are addressed; #9 is deliberately left open — episode/transcript reads are the core "browse the demo" feature and gating them behind the same API key as ingest would break the public demo experience, which is an acceptable, documented trade-off for a single-user portfolio project.
- **Placeholder scan:** no TBD/TODO markers; every step has literal code.
- **Type/name consistency:** `assert_safe_url` signature is identical everywhere it's imported (Task 1 defines it, Task 1 alone calls it). `MAX_UPLOAD_BYTES`/`is_probably_audio`/`read_capped` names match between `validators.py` (Task 2) and every router/test usage in Tasks 2–4. `require_api_key`/`limiter` names match between `core/auth.py`, `core/rate_limit.py` (Task 4) and every router that imports them. The `ChatRequest`/`TTSRequest` body parameter rename from `request` to `body` (Task 4) is applied consistently to both the signature and every internal reference (`body.message`, `body.thread_id`, `body.text`) in the same task's diff.
