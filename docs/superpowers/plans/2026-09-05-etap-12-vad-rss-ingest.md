# Etap 12 — VAD + Podcast RSS Batch Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit convention override for this plan:** Do NOT commit after individual tasks. Implementer subagents must leave all changes staged/unstaged. ONE combined commit is made at the very end, after all tasks pass review (per project convention: 1 squash-commit per etap on `main`).

**Goal:** Add two independent voice-AI/ingest quality-of-life features: (1) client-side voice activity detection (VAD) that auto-stops microphone recording when the user stops speaking, and (2) batch ingestion of an entire podcast RSS feed in one request instead of ingesting episodes one URL at a time.

**Architecture:** VAD is implemented client-side only, using the Web Audio API's `AnalyserNode` to compute RMS amplitude on an animation-frame loop and auto-triggering the existing `stop()` recording flow after a silence timeout — no new dependencies, no backend changes. RSS ingest adds a `feedparser`-based parser (`backend/app/ingest/rss.py`) that extracts episode audio-enclosure URLs from a feed, and a batch service function that reuses the existing `ingest_from_url` pipeline per episode, exposed via a new `POST /ingest/rss` endpoint and a third tab in the existing `FileUploader` UI.

**Tech Stack:** `feedparser` (pure-Python RSS/Atom parsing, no C extensions — safe on Windows), Web Audio API (`AudioContext`, `AnalyserNode`, `requestAnimationFrame`), existing `ingest_from_url` / `MediaRecorder` / `FileUploader` infrastructure.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/ingest/rss.py` | Create | `parse_feed(url, limit)` — pure function, extracts `{title, audio_url}` from RSS entries |
| `backend/tests/test_rss.py` | Create | Unit tests for `parse_feed` (mocked `feedparser.parse`) |
| `backend/app/ingest/service.py` | Modify | Add `ingest_from_rss(feed_url, db, limit)` |
| `backend/tests/test_ingest_service.py` | Modify or create | Tests for `ingest_from_rss` (mocked `parse_feed` + `ingest_from_url`) |
| `backend/app/ingest/router.py` | Modify | Add `POST /ingest/rss` |
| `backend/tests/test_ingest_router.py` | Modify or create | Tests for the new endpoint |
| `backend/requirements.txt` | Modify | Add `feedparser` |
| `frontend/components/FileUploader.tsx` | Modify | Add "RSS Feed" tab |
| `frontend/components/AudioRecorder.tsx` | Modify | Add energy-based VAD auto-stop |

---

### Task 1: RSS feed parser (pure function, unit-tested)

**Files:**
- Create: `backend/app/ingest/rss.py`
- Create: `backend/tests/test_rss.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_rss.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd backend
python -m pytest tests/test_rss.py -v
```

Expected: ImportError — `app.ingest.rss` doesn't exist yet.

- [ ] **Step 3: Add `feedparser` to requirements and install it**

Add to `backend/requirements.txt`:
```
feedparser==6.0.11
```

Install in whatever environment tests run against (check how prior tasks in this project installed packages — likely `pip install feedparser==6.0.11` against the active Python, e.g. `C:\Python311\python.exe -m pip install feedparser==6.0.11`).

- [ ] **Step 4: Implement `parse_feed`**

```python
# backend/app/ingest/rss.py
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
```

- [ ] **Step 5: Run tests — should pass**

```
python -m pytest tests/test_rss.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 6: Run full suite to check for regressions**

```
python -m pytest tests/ -v
```

Expected: all existing tests + 5 new pass, no regressions.

- [ ] **Step 7: Do NOT commit** (per this plan's commit override — leave staged/unstaged).

---

### Task 2: `ingest_from_rss` service function + `POST /ingest/rss` endpoint

**Files:**
- Modify: `backend/app/ingest/service.py`
- Modify: `backend/app/ingest/router.py`
- Create: `backend/tests/test_ingest_rss.py`

**Context — current `backend/app/ingest/service.py` (Task 1's `rss.py` already exists, DO NOT MODIFY it):**

```python
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from .chunker import chunk_segments
from .embedder import embed_texts
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
```

**Current `backend/app/ingest/router.py`:**

```python
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .service import ingest_audio, ingest_from_url

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int


@router.post("/upload", response_model=EpisodeResponse)
def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
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
```

**Your task:**

Add `ingest_from_rss(feed_url, db, limit=5)` to `service.py` — it parses the feed, then calls `ingest_from_url` for each entry's audio URL, catching per-entry failures so one bad episode doesn't abort the batch. Add `POST /ingest/rss` to `router.py` that calls it and returns ingested episodes plus any errors.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_ingest_rss.py
import uuid
from unittest.mock import MagicMock, patch

import pytest

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

    with patch("app.ingest.service.parse_feed", return_value=entries), \
         patch("app.ingest.service.ingest_from_url", side_effect=fake_episodes) as mock_ingest:
        result = ingest_from_rss("https://example.com/feed.xml", db=MagicMock(), limit=5)

    assert len(result["episodes"]) == 2
    assert result["errors"] == []
    assert mock_ingest.call_count == 2
    mock_ingest.assert_any_call("https://example.com/ep1.mp3", MagicMock().__class__() if False else mock_ingest.call_args_list[0][0][1])


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
```

Fix the slightly awkward assertion in the first test — replace it with a cleaner check:

```python
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
```

Use this corrected version (replace the awkward one above) in the actual test file.

**Router test — add to `backend/tests/test_ingest_router.py`** (if this file doesn't exist yet, check for the actual existing ingest router test file name first — e.g. it might be `test_ingest_router.py` or similar; find it with a glob for `tests/test_ingest*.py` and add to whichever file already tests `/ingest/upload` and `/ingest/url`, matching its existing fixture/import style):

```python
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
    _, kwargs = mock_ingest.call_args
    assert kwargs.get("limit") == 5 or mock_ingest.call_args.args[-1] == 5
```

Adjust the exact assertion in the second test once you see how the router actually calls `ingest_from_rss` (positional vs keyword) — make it match.

- [ ] **Step 2: Run tests to confirm failure**

```
cd backend
python -m pytest tests/test_ingest_rss.py -v
```

Expected: ImportError / AttributeError — `ingest_from_rss` doesn't exist yet.

- [ ] **Step 3: Implement `ingest_from_rss` in `service.py`**

Add this import at the top of `backend/app/ingest/service.py`:
```python
from .rss import parse_feed
```

Add this function at the end of `service.py`:
```python
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

- [ ] **Step 4: Implement `POST /ingest/rss` in `router.py`**

Update the import line:
```python
from .service import ingest_audio, ingest_from_rss, ingest_from_url
```

Add near the other request models:
```python
class IngestRSSRequest(BaseModel):
    url: str
    limit: int = 5


class RSSErrorItem(BaseModel):
    title: str
    error: str


class RSSIngestResponse(BaseModel):
    episodes: list[EpisodeResponse]
    errors: list[RSSErrorItem]
```

Add the endpoint:
```python
@router.post("/rss", response_model=RSSIngestResponse)
def ingest_rss(body: IngestRSSRequest, db: Session = Depends(get_db)):
    result = ingest_from_rss(body.url, db, limit=body.limit)
    return RSSIngestResponse(
        episodes=[
            EpisodeResponse(id=str(ep.id), filename=ep.filename, chunk_count=len(ep.chunks))
            for ep in result["episodes"]
        ],
        errors=[RSSErrorItem(**err) for err in result["errors"]],
    )
```

- [ ] **Step 5: Run tests — should pass**

```
python -m pytest tests/test_ingest_rss.py -v
```

Expected: all new tests PASS (check the exact count you wrote — should be 4 service tests + 2 router tests, adjust if you placed router tests in a different existing file).

- [ ] **Step 6: Run full suite**

```
python -m pytest tests/ -v
```

Expected: no regressions.

- [ ] **Step 7: Do NOT commit.**

---

### Task 3: Frontend — RSS Feed tab in FileUploader

**Files:**
- Modify: `frontend/components/FileUploader.tsx`

No automated tests (UI component, no test harness in this project for React components — verification is `tsc --noEmit` + manual check in browser).

**Context — current `frontend/components/FileUploader.tsx` structure:** a `tab` state of `"file" | "url"`, a tab-toggle row, conditional rendering of drop-zone (file) or URL input (url), a shared `status` state (`idle | uploading | done | error`), and an episode list below fed by `loadEpisodes()`.

**Your task:** add a third tab `"rss"` with its own input (feed URL) and an episode-count limit selector, calling `POST /api/ingest/rss`, and showing a result summary (e.g. "Ingested 4 episodes, 1 failed").

- [ ] **Step 1: Extend the `tab` type and add RSS-specific state**

Change:
```tsx
const [tab, setTab] = useState<"file" | "url">("file");
```
to:
```tsx
const [tab, setTab] = useState<"file" | "url" | "rss">("file");
```

Add near the other state declarations:
```tsx
const [rssUrl, setRssUrl] = useState("");
const [rssLimit, setRssLimit] = useState(5);
```

- [ ] **Step 2: Add an `ingestRss` function** near `ingestUrl`:

```tsx
async function ingestRss() {
  const url = rssUrl.trim();
  if (!url) return;
  setStatus({ kind: "uploading", label: `Fetching feed & ingesting up to ${rssLimit} episodes…` });
  try {
    const res = await fetch("/api/ingest/rss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, limit: rssLimit }),
    });
    if (!res.ok) throw new Error(await res.text());
    const result: { episodes: IngestResult[]; errors: { title: string; error: string }[] } = await res.json();
    if (result.episodes.length === 0 && result.errors.length > 0) {
      setStatus({ kind: "error", message: `All ${result.errors.length} episode(s) failed to ingest.` });
    } else if (result.errors.length > 0) {
      setStatus({
        kind: "done",
        result: {
          id: "batch",
          filename: `${result.episodes.length} episode(s) ingested, ${result.errors.length} failed`,
          chunk_count: result.episodes.reduce((sum, e) => sum + e.chunk_count, 0),
        },
      });
    } else {
      setStatus({
        kind: "done",
        result: {
          id: "batch",
          filename: `${result.episodes.length} episode(s) ingested from feed`,
          chunk_count: result.episodes.reduce((sum, e) => sum + e.chunk_count, 0),
        },
      });
    }
    setRssUrl("");
    await loadEpisodes();
  } catch (err) {
    setStatus({ kind: "error", message: String(err) });
  }
}
```

- [ ] **Step 3: Add the tab button** — update the tab-toggle row (currently a 2-button flex row) to 3 buttons:

```tsx
<div className="flex text-xs rounded-lg overflow-hidden border border-gray-800">
  <button
    onClick={() => setTab("file")}
    className={`flex-1 py-1.5 font-medium transition-colors ${
      tab === "file" ? "bg-indigo-600 text-white" : "bg-gray-900 text-gray-400 hover:bg-gray-800"
    }`}
  >
    File
  </button>
  <button
    onClick={() => setTab("url")}
    className={`flex-1 py-1.5 font-medium transition-colors ${
      tab === "url" ? "bg-indigo-600 text-white" : "bg-gray-900 text-gray-400 hover:bg-gray-800"
    }`}
  >
    URL
  </button>
  <button
    onClick={() => setTab("rss")}
    className={`flex-1 py-1.5 font-medium transition-colors ${
      tab === "rss" ? "bg-indigo-600 text-white" : "bg-gray-900 text-gray-400 hover:bg-gray-800"
    }`}
  >
    RSS
  </button>
</div>
```

- [ ] **Step 4: Add the RSS panel** — insert after the existing `{tab === "url" && (...)}` block:

```tsx
{tab === "rss" && (
  <div className="flex flex-col gap-2">
    <input
      value={rssUrl}
      onChange={(e) => setRssUrl(e.target.value)}
      onKeyDown={(e) => e.key === "Enter" && !busy && ingestRss()}
      placeholder="https://…/podcast-feed.xml"
      disabled={busy}
      className="flex-1 min-w-0 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs placeholder-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
    />
    <div className="flex items-center gap-2">
      <label className="text-[10px] text-gray-500 whitespace-nowrap">Max episodes</label>
      <input
        type="number"
        min={1}
        max={20}
        value={rssLimit}
        onChange={(e) => setRssLimit(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
        disabled={busy}
        className="w-16 bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-xs disabled:opacity-50"
      />
      <button
        onClick={ingestRss}
        disabled={busy || !rssUrl.trim()}
        className="ml-auto shrink-0 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-3 py-1.5 rounded-lg text-xs font-semibold"
      >
        Ingest Feed
      </button>
    </div>
    <p className="text-[10px] text-gray-600">
      Downloads and transcribes up to {rssLimit} episode(s) from the feed. This can take several minutes.
    </p>
  </div>
)}
```

- [ ] **Step 5: TypeScript check**

```
cd C:\Users\simon\Voice-RAG-Agent\frontend
docker compose exec frontend sh -c "npx tsc --noEmit"
```

(If the stack isn't running, use `docker compose run --rm frontend sh -c "npm install && npx tsc --noEmit"` instead.)

Expected: no errors.

- [ ] **Step 6: Restart frontend container**

```
cd C:\Users\simon\Voice-RAG-Agent
docker compose restart frontend
```

- [ ] **Step 7: Do NOT commit.**

---

### Task 4: Frontend — energy-based VAD auto-stop in AudioRecorder

**Files:**
- Modify: `frontend/components/AudioRecorder.tsx`

No automated tests (Web Audio API requires a real microphone/browser). Verification: `tsc --noEmit` + manual test (speak, then go silent, confirm recording auto-stops after ~1.5s).

**Current file:**

```tsx
"use client";

import { useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

export default function AudioRecorder({ onTranscript }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        try {
          const res = await fetch("/api/voice/stt", { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          const { text } = await res.json();
          onTranscript(text);
        } catch (err) {
          setError(String(err));
        }
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
    } catch (err) {
      setError(String(err));
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={recording ? stop : start}
        className={`px-6 py-3 rounded-full font-semibold transition-colors ${
          recording
            ? "bg-red-600 hover:bg-red-700 animate-pulse"
            : "bg-indigo-600 hover:bg-indigo-700"
        }`}
      >
        {recording ? "Stop Recording" : "Ask with Voice"}
      </button>
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}
```

**Design:** on `start()`, in addition to the existing `MediaRecorder`, create an `AudioContext` + `AnalyserNode` fed by the same microphone stream. On each animation frame, compute RMS amplitude from `getByteTimeDomainData`. Track when speech first crosses the threshold; once speech has been detected for at least `MIN_SPEECH_MS`, start counting continuous silence — if silence exceeds `SILENCE_TIMEOUT_MS`, call `stop()` automatically. Clean up the analysis `AudioContext` (separate from the recording `MediaRecorder`) whenever recording stops, including on unmount.

**Exact replacement file:**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

const SILENCE_THRESHOLD = 0.015; // RMS amplitude below this counts as silence
const SILENCE_TIMEOUT_MS = 1500; // auto-stop after this much continuous silence once speech has started
const MIN_SPEECH_MS = 300; // ignore silence detection until at least this much speech has been heard

export default function AudioRecorder({ onTranscript }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vadStatus, setVadStatus] = useState<"idle" | "listening" | "silence">("idle");
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechStartedAtRef = useRef<number | null>(null);

  function stopVadLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    audioCtxRef.current = null;
    analyserRef.current = null;
    silenceStartRef.current = null;
    speechStartedAtRef.current = null;
  }

  function monitorSilence() {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - 128) / 128;
      sumSquares += normalized * normalized;
    }
    const rms = Math.sqrt(sumSquares / data.length);

    const now = performance.now();
    if (speechStartedAtRef.current === null && rms > SILENCE_THRESHOLD) {
      speechStartedAtRef.current = now;
    }
    const speechElapsed = speechStartedAtRef.current !== null ? now - speechStartedAtRef.current : 0;

    if (rms > SILENCE_THRESHOLD) {
      silenceStartRef.current = null;
      setVadStatus("listening");
    } else if (speechStartedAtRef.current !== null && speechElapsed > MIN_SPEECH_MS) {
      if (silenceStartRef.current === null) {
        silenceStartRef.current = now;
      }
      setVadStatus("silence");
      if (now - silenceStartRef.current > SILENCE_TIMEOUT_MS) {
        stop();
        return;
      }
    }

    rafRef.current = requestAnimationFrame(monitorSilence);
  }

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stopVadLoop();
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        try {
          const res = await fetch("/api/voice/stt", { method: "POST", body: form });
          if (!res.ok) throw new Error(await res.text());
          const { text } = await res.json();
          onTranscript(text);
        } catch (err) {
          setError(String(err));
        }
        setVadStatus("idle");
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      setVadStatus("listening");

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      silenceStartRef.current = null;
      speechStartedAtRef.current = null;
      rafRef.current = requestAnimationFrame(monitorSilence);
    } catch (err) {
      setError(String(err));
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  useEffect(() => {
    return () => stopVadLoop();
  }, []);

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={recording ? stop : start}
        className={`px-6 py-3 rounded-full font-semibold transition-colors ${
          recording
            ? "bg-red-600 hover:bg-red-700 animate-pulse"
            : "bg-indigo-600 hover:bg-indigo-700"
        }`}
      >
        {recording ? "Stop Recording" : "Ask with Voice"}
      </button>
      {recording && (
        <p className="text-xs text-gray-500">
          {vadStatus === "silence" ? "Silence detected — stopping soon…" : "Listening…"}
        </p>
      )}
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 1: Replace `frontend/components/AudioRecorder.tsx`** with the exact content above.

- [ ] **Step 2: TypeScript check**

```
cd C:\Users\simon\Voice-RAG-Agent\frontend
docker compose exec frontend sh -c "npx tsc --noEmit"
```

Expected: no errors.

- [ ] **Step 3: Restart frontend container**

```
cd C:\Users\simon\Voice-RAG-Agent
docker compose restart frontend
```

- [ ] **Step 4: Manual verification note in report** — since no real microphone is available to the agent, state clearly in the report that this was verified via type-check + code review only, and that live microphone testing is a manual step for the user (speak a sentence, stop talking, confirm the button auto-flips from "Stop Recording" back to idle after ~1.5s of silence, and that the transcript still arrives normally).

- [ ] **Step 5: Do NOT commit.**

---

## Final Verification (controller, not a subagent task)

After all 4 tasks complete and pass review:

```bash
cd backend
python -m pytest tests/ -v
```

Expected: all existing tests + new RSS tests pass, 0 regressions.

```bash
cd frontend
docker compose exec frontend sh -c "npx tsc --noEmit"
```

Expected: 0 errors.

Then add a `DECISIONS.md` entry for the two key choices (energy-based VAD vs. Silero/WebRTC VAD; `feedparser` vs. hand-rolled XML parsing), and make ONE combined commit for the whole Etap 12 stage:

```bash
git add <all etap-12 files>
git commit -m "feat(etap-12): energy-based VAD auto-stop + podcast RSS batch ingest"
```

(No `Co-Authored-By` trailer — current session convention omits it entirely.)

---

## Self-Review

**Spec coverage:**
- ✅ VAD: auto-stop recording when speech ends (Task 4 — energy-based, `AnalyserNode` + silence timeout)
- ✅ RSS URL ingestion: parse feed, download + ingest all episodes automatically (Tasks 1–3 — `parse_feed`, `ingest_from_rss`, `POST /ingest/rss`, FileUploader RSS tab)

**No placeholders found.**

**Type consistency:** `parse_feed(feed_url, limit)` returns `list[dict]` with keys `title`/`audio_url` — consumed in `ingest_from_rss` via `entry["audio_url"]` / `entry["title"]` ✅. `ingest_from_rss` returns `{"episodes": list[Episode], "errors": list[dict]}` — consumed in router via `result["episodes"]` / `result["errors"]` ✅. Frontend `IngestResult` type (already defined in `FileUploader.tsx` for the existing upload/url flows) is reused for the RSS summary's synthetic `id: "batch"` entry — no new type needed.
