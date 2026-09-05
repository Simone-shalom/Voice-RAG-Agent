# Etap 1 — Ingest Pipeline + Baza Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload an audio file (or provide a URL), transcribe it with Whisper, chunk by natural speech boundaries, embed chunks, store in PostgreSQL/pgvector, and expose a semantic search endpoint returning chunks with timestamps.

**Architecture:** Two new FastAPI modules — `ingest/` (upload → transcribe → chunk → embed → save) and `search/` (query → embed → vector similarity → return chunks with timestamps). SQLAlchemy models `Episode` and `Chunk` with a pgvector `Vector(1536)` column. All external API calls are isolated in thin wrappers (`transcriber.py`, `embedder.py`) so they're trivial to mock in tests.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x, psycopg[binary], pgvector, openai SDK (Whisper API + `text-embedding-3-small`), httpx, pytest, pytest-mock

---

## File Map

**Create:**
- `backend/app/db/__init__.py`
- `backend/app/db/models.py` — SQLAlchemy ORM models (Episode, Chunk)
- `backend/app/db/session.py` — engine + SessionLocal + get_db dependency + init_db
- `backend/app/ingest/__init__.py`
- `backend/app/ingest/chunker.py` — pure function: Whisper segments → chunks with timestamps
- `backend/app/ingest/transcriber.py` — OpenAI Whisper API thin wrapper
- `backend/app/ingest/embedder.py` — OpenAI `text-embedding-3-small` thin wrapper
- `backend/app/ingest/service.py` — orchestrates transcribe → chunk → embed → save
- `backend/app/ingest/router.py` — `POST /ingest/upload`, `POST /ingest/url`
- `backend/app/search/__init__.py`
- `backend/app/search/searcher.py` — pgvector cosine similarity search
- `backend/app/search/router.py` — `GET /search`
- `backend/tests/__init__.py`
- `backend/tests/conftest.py` — shared fixtures (TestClient, mock DB override)
- `backend/tests/test_models.py`
- `backend/tests/test_session.py`
- `backend/tests/test_chunker.py`
- `backend/tests/test_transcriber.py`
- `backend/tests/test_embedder.py`
- `backend/tests/test_ingest_service.py`
- `backend/tests/test_ingest_router.py`
- `backend/tests/test_searcher.py`
- `backend/tests/test_search_router.py`
- `backend/tests/test_main.py`

**Modify:**
- `backend/requirements.txt` — add new deps
- `backend/app/main.py` — mount routers, init DB on startup

---

## Task 1: Update dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Replace requirements.txt content**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
sqlalchemy==2.0.36
psycopg[binary]==3.2.3
pgvector==0.3.5
openai==1.54.3
httpx==0.27.2
python-multipart==0.0.12
pytest==8.3.3
pytest-mock==3.14.0
```

- [ ] **Step 2: Verify Docker build**

```bash
docker compose build backend
```
Expected: build completes without errors (Dockerfile already copies `requirements.txt` and runs `pip install`, no changes needed there — `psycopg[binary]` bundles its own libpq, no system packages required).

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore(etap-1): add dependencies for ingest pipeline"
```

---

## Task 2: DB models

**Files:**
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/models.py`
- Create: `backend/tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_models.py`:
```python
from app.db.models import Episode, Chunk


def test_episode_has_required_columns():
    cols = {c.name for c in Episode.__table__.columns}
    assert {"id", "filename", "source_url", "created_at"} <= cols


def test_chunk_has_required_columns():
    cols = {c.name for c in Chunk.__table__.columns}
    assert {"id", "episode_id", "start_ts", "end_ts", "text", "embedding"} <= cols


def test_chunk_embedding_dimension():
    embedding_col = Chunk.__table__.columns["embedding"]
    assert embedding_col.type.dim == 1536
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd backend && python -m pytest tests/test_models.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`.

- [ ] **Step 3: Create `backend/app/db/__init__.py`** (empty file)

- [ ] **Step 4: Create `backend/app/db/models.py`**

```python
import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    chunks = relationship("Chunk", back_populates="episode", lazy="select")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(UUID(as_uuid=True), ForeignKey("episodes.id"), nullable=False)
    start_ts = Column(Float, nullable=False)
    end_ts = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    episode = relationship("Episode", back_populates="chunks")
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_models.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/db/ backend/tests/test_models.py
git commit -m "feat(etap-1): add Episode and Chunk SQLAlchemy models"
```

---

## Task 3: DB session + pgvector init

**Files:**
- Create: `backend/app/db/session.py`
- Create: `backend/tests/test_session.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_session.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

import inspect
from app.db.session import get_db, SessionLocal, init_db


def test_get_db_is_generator():
    assert inspect.isgeneratorfunction(get_db)


def test_session_local_exists():
    assert SessionLocal is not None


def test_init_db_is_callable():
    assert callable(init_db)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd backend && python -m pytest tests/test_session.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db.session'`.

- [ ] **Step 3: Create `backend/app/db/session.py`**

```python
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(engine_):
    """Enable pgvector extension and create all tables."""
    from .models import Base

    with engine_.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine_)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_session.py -v
```
Expected: 3 passed (no real DB connection needed — `create_engine` is lazy).

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/session.py backend/tests/test_session.py
git commit -m "feat(etap-1): add DB session factory and pgvector init helper"
```

---

## Task 4: Chunker

**Files:**
- Create: `backend/app/ingest/__init__.py`
- Create: `backend/app/ingest/chunker.py`
- Create: `backend/tests/test_chunker.py`

This is the most critical logic in Etap 1 — chunks must align to speech boundaries, never character counts, and `start_ts`/`end_ts` must stay accurate for citation.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_chunker.py`:
```python
from app.ingest.chunker import chunk_segments


def make_seg(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_empty_input_returns_empty():
    assert chunk_segments([]) == []


def test_single_segment_returns_one_chunk():
    segs = [make_seg("Hello world.", 0.0, 1.5)]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 1.5
    assert "Hello world" in chunks[0]["text"]


def test_pause_splits_into_two_chunks():
    segs = [
        make_seg("First sentence.", 0.0, 2.0),
        make_seg("Second sentence.", 3.0, 5.0),  # pause = 1.0s, above threshold
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 2
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 2.0
    assert chunks[1]["start_ts"] == 3.0
    assert chunks[1]["end_ts"] == 5.0


def test_no_pause_keeps_in_one_chunk():
    segs = [
        make_seg("First.", 0.0, 1.0),
        make_seg("Second.", 1.2, 2.0),  # pause = 0.2s, below threshold
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 2.0


def test_chunk_text_has_no_leading_trailing_whitespace():
    segs = [
        make_seg("Alpha beta.", 0.0, 2.0),
        make_seg("Gamma delta.", 3.0, 5.0),
    ]
    chunks = chunk_segments(segs)
    for chunk in chunks:
        assert chunk["text"] == chunk["text"].strip()


def test_word_count_limit_splits_long_chunk():
    # 30 segments of 15 words each = 450 words, above MAX_CHUNK_WORDS=300, pauses all small
    segs = [make_seg(" ".join(["word"] * 15), i * 1.1, i * 1.1 + 1.0) for i in range(30)]
    chunks = chunk_segments(segs)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk["text"].split()) <= 350  # cap plus one segment of slack
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_chunker.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest'`.

- [ ] **Step 3: Create `backend/app/ingest/__init__.py`** (empty file)

- [ ] **Step 4: Create `backend/app/ingest/chunker.py`**

```python
PAUSE_THRESHOLD = 0.5   # seconds — pauses longer than this trigger a chunk boundary
MAX_CHUNK_WORDS = 300    # hard cap to prevent runaway chunks


def chunk_segments(segments: list[dict]) -> list[dict]:
    """
    Split Whisper segments into chunks at natural speech boundaries.

    A boundary occurs when:
    - the pause between two consecutive segments exceeds PAUSE_THRESHOLD, OR
    - the accumulated word count hits MAX_CHUNK_WORDS.

    Each returned chunk has: {start_ts, end_ts, text}
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = segments[0]["start"]
    current_words = 0

    for i, seg in enumerate(segments):
        seg_text = seg["text"].strip()
        current_texts.append(seg_text)
        current_words += len(seg_text.split())

        is_last = i == len(segments) - 1
        next_seg = segments[i + 1] if not is_last else None
        pause = (next_seg["start"] - seg["end"]) if next_seg else 0.0

        should_split = is_last or pause >= PAUSE_THRESHOLD or current_words >= MAX_CHUNK_WORDS

        if should_split:
            chunks.append(
                {
                    "start_ts": current_start,
                    "end_ts": seg["end"],
                    "text": " ".join(current_texts),
                }
            )
            if not is_last:
                current_texts = []
                current_start = next_seg["start"]
                current_words = 0

    return chunks
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_chunker.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ingest/ backend/tests/test_chunker.py
git commit -m "feat(etap-1): implement natural-boundary chunker with timestamp preservation"
```

---

## Task 5: Transcriber

**Files:**
- Create: `backend/app/ingest/transcriber.py`
- Create: `backend/tests/test_transcriber.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_transcriber.py`:
```python
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.ingest.transcriber import transcribe


def make_mock_response():
    seg1 = MagicMock(text="Hello world.", start=0.0, end=2.0)
    seg2 = MagicMock(text="This is a test.", start=2.5, end=4.0)
    response = MagicMock()
    response.segments = [seg1, seg2]
    return response


def test_transcribe_returns_segment_list(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("app.ingest.transcriber.client.audio.transcriptions.create", return_value=make_mock_response()):
        result = transcribe(audio_file)

    assert len(result) == 2
    assert result[0] == {"text": "Hello world.", "start": 0.0, "end": 2.0}


def test_transcribe_requests_word_and_segment_timestamps(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("app.ingest.transcriber.client.audio.transcriptions.create") as mock_create:
        mock_create.return_value = make_mock_response()
        transcribe(audio_file)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["timestamp_granularities"] == ["word", "segment"]
    assert call_kwargs["response_format"] == "verbose_json"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_transcriber.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest.transcriber'`.

- [ ] **Step 3: Create `backend/app/ingest/transcriber.py`**

```python
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def transcribe(audio_path: Path) -> list[dict]:
    """
    Transcribe audio using the OpenAI Whisper API with word-level timestamps.
    Returns list of segment dicts: {text, start, end}.
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in response.segments]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_transcriber.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/transcriber.py backend/tests/test_transcriber.py
git commit -m "feat(etap-1): add Whisper transcriber with word-level timestamps"
```

---

## Task 6: Embedder

**Files:**
- Create: `backend/app/ingest/embedder.py`
- Create: `backend/tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_embedder.py`:
```python
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.ingest.embedder import embed_texts


def make_mock_embedding_response(n: int):
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
    return response


def test_embed_texts_returns_list_of_vectors():
    with patch("app.ingest.embedder.client.embeddings.create", return_value=make_mock_embedding_response(2)):
        result = embed_texts(["Hello world.", "Another sentence."])

    assert len(result) == 2
    assert len(result[0]) == 1536


def test_embed_texts_uses_correct_model():
    with patch("app.ingest.embedder.client.embeddings.create") as mock_create:
        mock_create.return_value = make_mock_embedding_response(1)
        embed_texts(["test"])

    assert mock_create.call_args.kwargs["model"] == "text-embedding-3-small"


def test_embed_empty_list_returns_empty():
    assert embed_texts([]) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_embedder.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest.embedder'`.

- [ ] **Step 3: Create `backend/app/ingest/embedder.py`**

```python
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI text-embedding-3-small. Returns one 1536-dim vector per text."""
    if not texts:
        return []
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_embedder.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/embedder.py backend/tests/test_embedder.py
git commit -m "feat(etap-1): add OpenAI text-embedding-3-small embedder"
```

---

## Task 7: Ingest service

**Files:**
- Create: `backend/app/ingest/service.py`
- Create: `backend/tests/test_ingest_service.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_ingest_service.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.ingest.service import ingest_audio


def make_mock_db():
    db = MagicMock()
    return db


def test_ingest_audio_creates_episode_and_chunks(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    mock_segments = [
        {"text": "Hello world.", "start": 0.0, "end": 2.0},
        {"text": "Another sentence.", "start": 3.0, "end": 5.0},  # pause > 0.5s -> 2 chunks
    ]
    mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

    with patch("app.ingest.service.transcribe", return_value=mock_segments) as mock_t, \
         patch("app.ingest.service.embed_texts", return_value=mock_embeddings) as mock_e:
        ingest_audio(audio_file, "test.mp3", None, db)

    mock_t.assert_called_once_with(audio_file)
    mock_e.assert_called_once()
    assert db.add.call_count == 3  # 1 episode + 2 chunks
    db.commit.assert_called_once()


def test_ingest_audio_sets_episode_filename_and_source(tmp_path):
    audio_file = tmp_path / "my_podcast.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]):
        ingest_audio(audio_file, "my_podcast.mp3", "http://example.com/audio.mp3", db)

    added_episode = db.add.call_args_list[0][0][0]
    assert added_episode.filename == "my_podcast.mp3"
    assert added_episode.source_url == "http://example.com/audio.mp3"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_ingest_service.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest.service'`.

- [ ] **Step 3: Create `backend/app/ingest/service.py`**

```python
import tempfile
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from .chunker import chunk_segments
from .embedder import embed_texts
from .transcriber import transcribe
from ..db.models import Chunk, Episode


def ingest_audio(audio_path: Path, filename: str, source_url: str | None, db: Session) -> Episode:
    episode = Episode(filename=filename, source_url=source_url)
    db.add(episode)
    db.flush()  # populate episode.id without committing

    segments = transcribe(audio_path)
    chunk_dicts = chunk_segments(segments)
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)

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
    with httpx.Client(follow_redirects=True, timeout=60.0) as http:
        response = http.get(url)
        response.raise_for_status()

    suffix = Path(url).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        tmp_path = Path(f.name)

    try:
        return ingest_audio(tmp_path, Path(url).name or "audio", url, db)
    finally:
        tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_ingest_service.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/service.py backend/tests/test_ingest_service.py
git commit -m "feat(etap-1): add ingest service orchestrating transcribe to chunk to embed to save"
```

---

## Task 8: Ingest router

**Files:**
- Create: `backend/app/ingest/router.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_ingest_router.py`

- [ ] **Step 1: Create shared test fixtures**

`backend/tests/conftest.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# init_db would try to hit a real DB on import of app.main — patch it out for the whole test session.
with patch("app.db.session.init_db"):
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

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_ingest_router.py`:
```python
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
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_ingest_router.py -v
```
Expected: FAIL (404 — routers not mounted yet, or import error).

- [ ] **Step 4: Create `backend/app/ingest/router.py`**

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
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)
    try:
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    finally:
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

- [ ] **Step 5: Wire router into `backend/app/main.py` (temporary minimal wiring, finalized in Task 11)**

```python
from fastapi import FastAPI

from .ingest.router import router as ingest_router

app = FastAPI(title="Voice Knowledge Agent")
app.include_router(ingest_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_ingest_router.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/ingest/router.py backend/app/main.py backend/tests/conftest.py backend/tests/test_ingest_router.py
git commit -m "feat(etap-1): add ingest endpoints POST /ingest/upload and POST /ingest/url"
```

---

## Task 9: Semantic searcher

**Files:**
- Create: `backend/app/search/__init__.py`
- Create: `backend/app/search/searcher.py`
- Create: `backend/tests/test_searcher.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_searcher.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.search.searcher import semantic_search


def make_mock_row(chunk_id="c1", episode_id="e1", start_ts=1.0, end_ts=5.0, text="Hello", similarity=0.9):
    row = MagicMock()
    row.id = chunk_id
    row.episode_id = episode_id
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.similarity = similarity
    return row


def test_semantic_search_returns_chunk_dicts():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        make_mock_row(),
        make_mock_row(chunk_id="c2", similarity=0.8),
    ]

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        results = semantic_search("test query", limit=10, db=db)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity"] == 0.9
    assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity"}


def test_semantic_search_passes_limit_to_query():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        semantic_search("query", limit=5, db=db)

    _, params = db.execute.call_args[0]
    assert params["limit"] == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_searcher.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.search'`.

- [ ] **Step 3: Create `backend/app/search/__init__.py`** (empty file)

- [ ] **Step 4: Create `backend/app/search/searcher.py`**

```python
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..ingest.embedder import embed_texts


def semantic_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Embed the query and find the nearest chunks by cosine similarity.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}.
    """
    embedding = embed_texts([query])[0]
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text,
                   1 - (c.embedding <=> :emb::vector) AS similarity
            FROM chunks c
            ORDER BY c.embedding <=> :emb::vector
            LIMIT :limit
            """
        ),
        {"emb": embedding_str, "limit": limit},
    ).fetchall()

    return [
        {
            "chunk_id": str(r.id),
            "episode_id": str(r.episode_id),
            "start_ts": r.start_ts,
            "end_ts": r.end_ts,
            "text": r.text,
            "similarity": float(r.similarity),
        }
        for r in rows
    ]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_searcher.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/search/ backend/tests/test_searcher.py
git commit -m "feat(etap-1): add pgvector semantic searcher"
```

---

## Task 10: Search router

**Files:**
- Create: `backend/app/search/router.py`
- Create: `backend/tests/test_search_router.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_search_router.py`:
```python
from unittest.mock import patch

MOCK_RESULTS = [
    {
        "chunk_id": "aaa",
        "episode_id": "bbb",
        "start_ts": 10.0,
        "end_ts": 25.0,
        "text": "Relevant content here.",
        "similarity": 0.87,
    }
]


def test_search_returns_results(client):
    with patch("app.search.router.semantic_search", return_value=MOCK_RESULTS):
        response = client.get("/search?q=test+query")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["chunk_id"] == "aaa"
    assert body[0]["start_ts"] == 10.0


def test_search_missing_query_returns_422(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_search_passes_limit_to_searcher(client):
    with patch("app.search.router.semantic_search", return_value=[]) as mock_search:
        client.get("/search?q=hello&limit=5")

    assert mock_search.call_args.kwargs["limit"] == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_search_router.py -v
```
Expected: FAIL (404 — router not mounted yet).

- [ ] **Step 3: Create `backend/app/search/router.py`**

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db.session import get_db
from .searcher import semantic_search

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    return semantic_search(query=q, limit=limit, db=db)
```

- [ ] **Step 4: Mount router in `backend/app/main.py`**

Add to `backend/app/main.py`:
```python
from .search.router import router as search_router
```
and:
```python
app.include_router(search_router)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_search_router.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/search/router.py backend/app/main.py backend/tests/test_search_router.py
git commit -m "feat(etap-1): add GET /search endpoint"
```

---

## Task 11: Finalize main.py startup + full test run

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write the failing integration test**

`backend/tests/test_main.py`:
```python
def test_health_still_works(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_router_mounted(client):
    response = client.post("/ingest/upload")
    assert response.status_code == 422  # missing file, not 404


def test_search_router_mounted(client):
    response = client.get("/search")
    assert response.status_code == 422  # missing q, not 404
```

- [ ] **Step 2: Create `backend/tests/__init__.py`** (empty file)

- [ ] **Step 3: Run test to confirm current state**

```bash
cd backend && python -m pytest tests/test_main.py -v
```
Expected: 3 passed already (routers were mounted incrementally in Tasks 8 and 10) — this step is a checkpoint, not a red step.

- [ ] **Step 4: Finalize `backend/app/main.py`**

```python
from fastapi import FastAPI

from .db.session import engine, init_db
from .ingest.router import router as ingest_router
from .search.router import router as search_router

init_db(engine)

app = FastAPI(title="Voice Knowledge Agent")

app.include_router(ingest_router)
app.include_router(search_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 5: Run the full test suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all tests pass (28 total across Tasks 2-11).

- [ ] **Step 6: Manual smoke test — start the stack**

```bash
docker compose up -d
curl -s localhost:8000/health
```
Expected: `{"status":"ok"}`. This confirms `init_db` successfully connects to the real Postgres container and creates the `vector` extension + tables.

- [ ] **Step 7: Manual smoke test — verify tables exist**

```bash
docker compose exec postgres psql -U voicerag -d voicerag -c "\dt"
```
Expected: `episodes` and `chunks` tables listed.

- [ ] **Step 8: Commit**

```bash
git add backend/app/main.py backend/tests/__init__.py backend/tests/test_main.py
git commit -m "feat(etap-1): init pgvector extension and tables on backend startup"
```

---

## Definition of Done (manual verification — requires a real audio file and API keys)

Not automatable in this plan since it needs live API keys and a real audio file; perform once after Task 11 is merged:

```bash
# .env must have OPENAI_API_KEY set
docker compose up -d
curl -X POST localhost:8000/ingest/upload -F "file=@/path/to/20min-podcast.mp3"
# note the returned episode id and chunk_count
curl "localhost:8000/search?q=<topic mentioned in the recording>"
# manually listen to the source file at the returned start_ts of the top result — confirm it matches
```

Record the outcome (chunk_count, whether timestamps lined up, any surprises) in `DECISIONS.md` per the Etap 1 entry format, and write `docs/claude-md-fragments/etap-1.md` per the roadmap's step 7 (stack commands, chunk schema, upload command).

---

## Self-Review

**Spec coverage:**
- [x] Endpoint upload audio file → `POST /ingest/upload` (Task 8)
- [x] Endpoint ingest audio from URL → `POST /ingest/url` (Task 8)
- [x] Whisper transcription with word-level timestamps → `transcriber.py` requests `timestamp_granularities=["word","segment"]` (Task 5)
- [x] Chunking by natural speech boundaries (pause > 0.5s), not character count, `start_ts`/`end_ts` preserved → `chunker.py` (Task 4)
- [x] Regression test: chunk boundaries align with pauses, not mid-sentence cuts → `test_pause_splits_into_two_chunks`, `test_no_pause_keeps_in_one_chunk` (Task 4)
- [x] Embeddings generated and saved to pgvector → `embedder.py` (Task 6) + `Chunk.embedding: Vector(1536)` (Task 2)
- [x] `GET /search?q=...` returns chunks + timestamps + similarity → `search/router.py` + `searcher.py` (Tasks 9-10)
- [x] pgvector extension enabled → `init_db` runs `CREATE EXTENSION IF NOT EXISTS vector` (Task 3)
- [x] `docker compose up` + `curl /health` still works → Task 11 manual smoke test

**Placeholder scan:** none found — every step has runnable code or an exact command.

**Type consistency:** `transcribe()` returns `list[dict]` shaped `{text, start, end}` (Task 5), consumed identically by `chunk_segments()` (Task 4) and `service.ingest_audio` (Task 7). `chunk_segments()` returns `list[dict]` shaped `{start_ts, end_ts, text}` (Task 4), consumed identically in `service.ingest_audio` (Task 7). `embed_texts()` returns `list[list[float]]` (Task 6), consumed identically in `service.ingest_audio` (Task 7) and `searcher.semantic_search` (Task 9). Router response field `chunk_count` matches `len(episode.chunks)` in both `EpisodeResponse` construction sites (Task 8).
