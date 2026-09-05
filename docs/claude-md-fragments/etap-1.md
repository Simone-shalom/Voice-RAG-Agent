# CLAUDE.md fragment — Etap 1: Ingest Pipeline

## Running tests

```bash
cd backend
python -m pytest tests/ -v          # 38 tests, all mocked (no real DB/API needed)
```

## Stack added in Etap 1

```
backend/requirements.txt additions:
  sqlalchemy==2.0.36, psycopg[binary]==3.2.3, pgvector==0.3.5
  openai==1.54.3, httpx==0.27.2, python-multipart==0.0.12
  pytest==8.3.3, pytest-mock==3.14.0
```

## Key modules

| Module | Purpose |
|--------|---------|
| `backend/app/db/models.py` | Episode + Chunk ORM models |
| `backend/app/db/session.py` | psycopg3 engine, `get_db` dependency (explicit rollback), `init_db` |
| `backend/app/ingest/chunker.py` | Speech-boundary chunker — pauses ≥ 0.5 s split, cap 300 words; drops whitespace-only segments |
| `backend/app/ingest/transcriber.py` | Whisper API with `timestamp_granularities=["word","segment"]`; lazy client |
| `backend/app/ingest/embedder.py` | `text-embedding-3-small` (1536 dim); lazy client |
| `backend/app/ingest/service.py` | Orchestrates transcribe → chunk → embed → save; streams URL downloads |
| `backend/app/ingest/router.py` | `POST /ingest/upload`, `POST /ingest/url` |
| `backend/app/search/searcher.py` | pgvector `<=>` cosine search; `WHERE embedding IS NOT NULL` |
| `backend/app/search/router.py` | `GET /search?q=...&limit=N` |

## Chunk schema (invariant going forward)

```
id UUID, episode_id UUID, start_ts FLOAT, end_ts FLOAT, text TEXT, embedding VECTOR(1536)
```

`start_ts`/`end_ts` are in seconds from audio start. Never break this — all downstream timestamp citations depend on it.

## Ingest command

```bash
# Requires OPENAI_API_KEY in .env
curl -X POST localhost:8000/ingest/upload -F "file=@/path/to/recording.mp3"
# Returns: {"id": "<uuid>", "filename": "...", "chunk_count": N}

curl "localhost:8000/search?q=topic+keyword&limit=5"
# Returns: [{chunk_id, episode_id, start_ts, end_ts, text, similarity}, ...]
```

## Test mocking pattern

OpenAI clients are lazy-initialized via `_client()` (lru_cache). Mock the getter:

```python
with patch("app.ingest.embedder._client", return_value=mock_openai_client):
    ...
with patch("app.ingest.transcriber._client", return_value=mock_openai_client):
    ...
```

## Decisions made (see DECISIONS.md for full rationale)

- Whisper API over faster-whisper (no local GPU needed, replaceable interface)
- Pause-boundary chunking over fixed character/token split (preserves timestamp accuracy)
- `text-embedding-3-small` over large (5× cheaper, sufficient quality)
- FastAPI lifespan for `init_db` (eliminates import-time DB race condition)
- 1 squash-commit per etap on `main` (clean git history mapping to project stages)
