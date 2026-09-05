# Etap 2 — Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `GET /search` with four modes — `semantic`, `bm25`, `hybrid` (RRF fusion), `hybrid+rerank` (Cohere Rerank) — and add a comparison script that runs all four modes side-by-side.

**Architecture:** Three new focused modules (`bm25.py`, `rrf.py`, `reranker.py`) added under `backend/app/search/`. `searcher.py` grows two new orchestration functions (`hybrid_search`, `hybrid_rerank_search`). The router gains a `?mode=` parameter and dispatches to the right function. A PostgreSQL `tsvector` generated column on `chunks` powers BM25 — added idempotently inside `init_db` so no separate migration tool is needed. All search functions return the same dict shape (with `similarity` as the generic score field) so the response model stays unchanged.

**Tech Stack:** PostgreSQL `tsvector`/`ts_rank`, pgvector cosine similarity, Reciprocal Rank Fusion (pure Python), Cohere Rerank API (`cohere==5.11.4`), pytest, pytest-mock.

---

## File Map

**Create:**
- `backend/app/search/bm25.py` — `bm25_search(query, limit, db) -> list[dict]`
- `backend/app/search/rrf.py` — `reciprocal_rank_fusion(result_lists, k=60) -> list[dict]` (pure, no I/O)
- `backend/app/search/reranker.py` — `rerank(query, chunks, top_n) -> list[dict]` (Cohere, lazy client)
- `backend/tests/test_bm25.py`
- `backend/tests/test_rrf.py`
- `backend/tests/test_reranker.py`
- `eval/__init__.py` (empty)
- `eval/search_comparison.py` — CLI comparison script (no unit tests; requires live DB + API keys)

**Modify:**
- `backend/requirements.txt` — add `cohere==5.11.4`
- `backend/app/db/session.py` — extend `init_db` with tsvector column + GIN index migration
- `backend/app/search/searcher.py` — add `hybrid_search`, `hybrid_rerank_search`
- `backend/app/search/router.py` — add `?mode=` parameter and dispatch logic
- `backend/tests/test_searcher.py` — add hybrid_search tests
- `backend/tests/test_search_router.py` — add mode-parameter tests

---

## Task 1: tsvector migration in init_db

**Files:**
- Modify: `backend/app/db/session.py`
- Modify: `backend/tests/test_session.py`

The `chunks.text_search` column is a PostgreSQL generated stored column (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`). PostgreSQL computes it for all existing rows when the column is added, so no backfill needed. The GIN index makes `@@` queries fast.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_session.py`:

```python
def test_init_db_sql_contains_tsvector():
    """init_db must run the tsvector migration — verify it calls execute at least twice."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("app.db.session.Base") as mock_base:
        init_db(mock_engine)

    # Should have been called 3+ times: CREATE EXTENSION, tsvector DO block, CREATE INDEX
    assert mock_conn.execute.call_count >= 3
```

Also add the missing import at the top of `test_session.py` (it currently only has `os` and `inspect`):

```python
from unittest.mock import MagicMock, patch
from app.db.session import get_db, SessionLocal, init_db
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd backend && python -m pytest tests/test_session.py::test_init_db_sql_contains_tsvector -v
```
Expected: FAIL — `init_db` only calls `execute` once (`CREATE EXTENSION vector`).

- [ ] **Step 3: Extend `init_db` in `backend/app/db/session.py`**

```python
def init_db(engine_):
    """Enable pgvector extension, create all tables, add tsvector search column."""
    from .models import Base

    with engine_.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'text_search'
                  ) THEN
                    ALTER TABLE chunks
                    ADD COLUMN text_search tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
                  END IF;
                END
                $$
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_text_search_gin "
                "ON chunks USING GIN(text_search)"
            )
        )
        conn.commit()
    Base.metadata.create_all(bind=engine_)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_session.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/session.py backend/tests/test_session.py
git commit -m "feat(etap-2): add tsvector generated column and GIN index migration to init_db"
```

---

## Task 2: Add cohere to requirements

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add cohere**

Append to `backend/requirements.txt`:
```
cohere==5.11.4
```

- [ ] **Step 2: Verify Docker build**

```bash
docker compose build backend
```
Expected: build completes without errors.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore(etap-2): add cohere SDK dependency"
```

---

## Task 3: BM25 searcher

**Files:**
- Create: `backend/app/search/bm25.py`
- Create: `backend/tests/test_bm25.py`

BM25 uses PostgreSQL's native full-text search. `plainto_tsquery` converts the query to a tsquery (AND of terms); `ts_rank` scores by term frequency. Returns same dict shape as `semantic_search` with `similarity` as the score field.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_bm25.py`:

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock
from app.search.bm25 import bm25_search


def make_mock_row(chunk_id="c1", episode_id="e1", start_ts=0.0, end_ts=5.0, text="Hello", bm25_score=0.5):
    row = MagicMock()
    row.id = chunk_id
    row.episode_id = episode_id
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.bm25_score = bm25_score
    return row


def test_bm25_search_returns_chunk_dicts():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        make_mock_row(),
        make_mock_row(chunk_id="c2", bm25_score=0.3),
    ]

    results = bm25_search("hello world", limit=10, db=db)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity"] == 0.5
    assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity"}


def test_bm25_search_passes_limit():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    bm25_search("query", limit=7, db=db)

    _, params = db.execute.call_args[0]
    assert params["limit"] == 7


def test_bm25_search_empty_results():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []

    results = bm25_search("no match", limit=10, db=db)

    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_bm25.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.search.bm25'`.

- [ ] **Step 3: Create `backend/app/search/bm25.py`**

```python
from sqlalchemy import text
from sqlalchemy.orm import Session


def bm25_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Full-text search using PostgreSQL tsvector/ts_rank.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}
    where similarity is the ts_rank score (higher = more relevant).
    """
    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text,
                   ts_rank(c.text_search, query) AS bm25_score
            FROM chunks c,
                 plainto_tsquery('english', :query_text) query
            WHERE c.text_search @@ query
            ORDER BY bm25_score DESC
            LIMIT :limit
            """
        ),
        {"query_text": query, "limit": limit},
    ).fetchall()

    return [
        {
            "chunk_id": str(r.id),
            "episode_id": str(r.episode_id),
            "start_ts": r.start_ts,
            "end_ts": r.end_ts,
            "text": r.text,
            "similarity": float(r.bm25_score),
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_bm25.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/bm25.py backend/tests/test_bm25.py
git commit -m "feat(etap-2): add BM25 searcher using PostgreSQL tsvector/ts_rank"
```

---

## Task 4: RRF fusion

**Files:**
- Create: `backend/app/search/rrf.py`
- Create: `backend/tests/test_rrf.py`

Pure function — no DB, no I/O. Takes multiple ranked lists (each already sorted best-first), assigns ordinal ranks (1, 2, 3...), computes `1/(k + rank)` per list, sums contributions, returns unified list sorted by total RRF score descending.

Critical: RRF fuses **ranks**, not raw scores. A chunk ranked #1 in BM25 gets the same contribution regardless of its raw ts_rank value.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_rrf.py`:

```python
from app.search.rrf import reciprocal_rank_fusion


def make_chunk(chunk_id, similarity=0.5):
    return {"chunk_id": chunk_id, "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
            "text": f"text {chunk_id}", "similarity": similarity}


def test_single_list_preserves_order():
    results = [make_chunk("a", 0.9), make_chunk("b", 0.5), make_chunk("c", 0.1)]
    fused = reciprocal_rank_fusion([results])
    assert [r["chunk_id"] for r in fused] == ["a", "b", "c"]


def test_chunk_in_both_lists_ranked_first():
    # "b" is in both lists; should outrank chunks appearing in only one list
    list1 = [make_chunk("a"), make_chunk("b")]
    list2 = [make_chunk("b"), make_chunk("c")]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0]["chunk_id"] == "b"


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_one_nonempty():
    results = [make_chunk("x"), make_chunk("y")]
    fused = reciprocal_rank_fusion([results, []])
    assert [r["chunk_id"] for r in fused] == ["x", "y"]


def test_result_contains_rrf_score():
    results = [make_chunk("a")]
    fused = reciprocal_rank_fusion([results])
    assert "rrf_score" in fused[0]
    assert isinstance(fused[0]["rrf_score"], float)


def test_rrf_uses_ranks_not_raw_scores():
    # "b" has lower raw similarity but is ranked #1 in list2 — RRF must not use similarity
    list1 = [make_chunk("a", similarity=0.99)]  # rank 1 in list1
    list2 = [make_chunk("b", similarity=0.01), make_chunk("a", similarity=0.98)]  # a is rank 2
    fused = reciprocal_rank_fusion([list1, list2])
    # "a": 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # "b": 1/(60+1) = 0.01639
    assert fused[0]["chunk_id"] == "a"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_rrf.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.search.rrf'`.

- [ ] **Step 3: Create `backend/app/search/rrf.py`**

```python
def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each list must be sorted best-first. Chunks are identified by 'chunk_id'.
    The same chunk appearing in multiple lists receives contributions from each.

    Returns a new list sorted by total RRF score descending, with 'rrf_score' added.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_data.setdefault(cid, chunk)

    return [
        {**chunk_data[cid], "rrf_score": score}
        for cid, score in sorted(scores.items(), key=lambda x: -x[1])
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_rrf.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/rrf.py backend/tests/test_rrf.py
git commit -m "feat(etap-2): add Reciprocal Rank Fusion (RRF) for hybrid search"
```

---

## Task 5: Cohere reranker

**Files:**
- Create: `backend/app/search/reranker.py`
- Create: `backend/tests/test_reranker.py`

Lazy Cohere client (same pattern as embedder/transcriber). If `COHERE_API_KEY` is not set, calling `rerank` raises `RuntimeError` — the router converts this to HTTP 400 with a clear message. Returns chunks reordered by Cohere's relevance score with `similarity` as the score field.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_reranker.py`:

```python
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
import pytest
from app.search.reranker import rerank


def make_chunks(n=3):
    return [
        {"chunk_id": chr(ord("a") + i), "episode_id": "ep1",
         "start_ts": float(i), "end_ts": float(i + 5),
         "text": f"content {chr(ord('a') + i)}", "similarity": 0.5}
        for i in range(n)
    ]


def make_mock_client(reranked_indices: list[int]):
    mock = MagicMock()
    mock.rerank.return_value.results = [
        MagicMock(index=i, relevance_score=1.0 - rank * 0.1)
        for rank, i in enumerate(reranked_indices)
    ]
    return mock


def test_rerank_reorders_chunks():
    chunks = make_chunks(3)  # a, b, c
    mock_client = make_mock_client([2, 0, 1])  # Cohere says c > a > b

    with patch("app.search.reranker._client", return_value=mock_client):
        result = rerank("query", chunks, top_n=3)

    assert result[0]["chunk_id"] == "c"
    assert result[1]["chunk_id"] == "a"
    assert result[2]["chunk_id"] == "b"


def test_rerank_sets_similarity_to_relevance_score():
    chunks = make_chunks(1)
    mock_client = make_mock_client([0])
    mock_client.rerank.return_value.results[0].relevance_score = 0.92

    with patch("app.search.reranker._client", return_value=mock_client):
        result = rerank("query", chunks, top_n=1)

    assert result[0]["similarity"] == pytest.approx(0.92)


def test_rerank_empty_chunks_returns_empty():
    result = rerank("query", [], top_n=5)
    assert result == []


def test_rerank_raises_if_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        # Clear lru_cache so it re-reads the env
        from app.search import reranker
        reranker._client.cache_clear()
        with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
            rerank("query", make_chunks(1), top_n=1)
    # Restore: re-clear cache so later tests use fresh state
    reranker._client.cache_clear()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_reranker.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.search.reranker'`.

- [ ] **Step 3: Create `backend/app/search/reranker.py`**

```python
import os
from functools import lru_cache

import cohere


@lru_cache(maxsize=1)
def _client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set — cannot use hybrid+rerank mode")
    return cohere.Client(api_key=api_key)


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Rerank chunks using Cohere Rerank API.
    Returns chunks sorted by Cohere's relevance score with 'similarity' updated.
    Raises RuntimeError if COHERE_API_KEY is not set.
    """
    if not chunks:
        return []
    texts = [c["text"] for c in chunks]
    response = _client().rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=min(top_n, len(chunks)),
    )
    return [
        {**chunks[r.index], "similarity": float(r.relevance_score)}
        for r in response.results
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_reranker.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/reranker.py backend/tests/test_reranker.py
git commit -m "feat(etap-2): add Cohere Rerank wrapper with lazy client init"
```

---

## Task 6: hybrid_search and hybrid_rerank_search in searcher.py

**Files:**
- Modify: `backend/app/search/searcher.py`
- Modify: `backend/tests/test_searcher.py`

`hybrid_search` fetches 2× limit from both semantic and BM25, fuses via RRF, returns top `limit`. The extra fetch headroom ensures that chunks present in only one result set still make it into the final list before truncation.

`hybrid_rerank_search` takes the hybrid result (2× limit) and passes it to the Cohere reranker.

Both functions return the same dict shape as `semantic_search` — `similarity` is the relevant score for that mode.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_searcher.py`:

```python
from app.search.searcher import semantic_search, hybrid_search, hybrid_rerank_search


def test_hybrid_search_fuses_semantic_and_bm25():
    sem = [{"chunk_id": "a", "episode_id": "e1", "start_ts": 0.0, "end_ts": 5.0,
            "text": "alpha", "similarity": 0.9},
           {"chunk_id": "b", "episode_id": "e1", "start_ts": 5.0, "end_ts": 10.0,
            "text": "beta", "similarity": 0.7}]
    bm25 = [{"chunk_id": "b", "episode_id": "e1", "start_ts": 5.0, "end_ts": 10.0,
             "text": "beta", "similarity": 0.8},
            {"chunk_id": "c", "episode_id": "e1", "start_ts": 10.0, "end_ts": 15.0,
             "text": "gamma", "similarity": 0.6}]
    db = MagicMock()

    with patch("app.search.searcher.semantic_search", return_value=sem) as m_sem, \
         patch("app.search.searcher.bm25_search", return_value=bm25) as m_bm25:
        result = hybrid_search("test query", limit=3, db=db)

    # both sub-searchers called with 2× limit
    assert m_sem.call_args.kwargs["limit"] == 6
    assert m_bm25.call_args.kwargs["limit"] == 6
    # "b" appears in both lists → highest RRF score → first
    assert result[0]["chunk_id"] == "b"
    # all three distinct chunks returned
    assert {r["chunk_id"] for r in result} == {"a", "b", "c"}
    # similarity field present (mapped from rrf_score)
    assert "similarity" in result[0]


def test_hybrid_search_returns_at_most_limit():
    sem = [{"chunk_id": str(i), "episode_id": "e1", "start_ts": 0.0, "end_ts": 1.0,
            "text": "t", "similarity": 0.5} for i in range(20)]
    db = MagicMock()

    with patch("app.search.searcher.semantic_search", return_value=sem), \
         patch("app.search.searcher.bm25_search", return_value=[]):
        result = hybrid_search("q", limit=5, db=db)

    assert len(result) <= 5


def test_hybrid_rerank_search_calls_reranker():
    fused = [{"chunk_id": "x", "episode_id": "e1", "start_ts": 0.0, "end_ts": 5.0,
              "text": "content x", "similarity": 0.5}]
    reranked = [{**fused[0], "similarity": 0.95}]
    db = MagicMock()

    with patch("app.search.searcher.hybrid_search", return_value=fused), \
         patch("app.search.searcher.rerank", return_value=reranked):
        result = hybrid_rerank_search("query", limit=5, db=db)

    assert result[0]["similarity"] == 0.95
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_searcher.py -v
```
Expected: 3 new tests FAIL with `ImportError` (functions not defined yet).

- [ ] **Step 3: Update `backend/app/search/searcher.py`**

```python
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..ingest.embedder import embed_texts
from .bm25 import bm25_search
from .reranker import rerank
from .rrf import reciprocal_rank_fusion


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
            WHERE c.embedding IS NOT NULL
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


def hybrid_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Fuse semantic and BM25 results via Reciprocal Rank Fusion.
    Fetches 2× limit from each sub-searcher for fusion headroom.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}
    where similarity is the RRF score.
    """
    fetch = limit * 2
    sem = semantic_search(query, limit=fetch, db=db)
    bm25 = bm25_search(query, limit=fetch, db=db)
    fused = reciprocal_rank_fusion([sem, bm25])
    # Map rrf_score → similarity for uniform response shape
    return [
        {**chunk, "similarity": chunk["rrf_score"]}
        for chunk in fused[:limit]
    ]


def hybrid_rerank_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Hybrid RRF search followed by Cohere reranking.
    Raises RuntimeError if COHERE_API_KEY is not set.
    """
    candidates = hybrid_search(query, limit=limit * 2, db=db)
    return rerank(query, candidates, top_n=limit)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_searcher.py -v
```
Expected: 5 passed (2 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add backend/app/search/searcher.py backend/tests/test_searcher.py
git commit -m "feat(etap-2): add hybrid_search (RRF) and hybrid_rerank_search to searcher"
```

---

## Task 7: Update search router with mode parameter

**Files:**
- Modify: `backend/app/search/router.py`
- Modify: `backend/tests/test_search_router.py`

The `?mode=` parameter selects which search function runs. Default is `semantic` to preserve backward compatibility. Invalid mode returns 422 (FastAPI handles this via `Literal` annotation). `hybrid+rerank` without `COHERE_API_KEY` returns 400 with the RuntimeError message.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_search_router.py`:

```python
MOCK_HYBRID_RESULTS = [
    {
        "chunk_id": "hhh",
        "episode_id": "eee",
        "start_ts": 5.0,
        "end_ts": 20.0,
        "text": "Hybrid result.",
        "similarity": 0.75,
    }
]


def test_search_defaults_to_semantic_mode(client):
    with patch("app.search.router.semantic_search", return_value=MOCK_RESULTS) as m:
        response = client.get("/search?q=hello")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_bm25_mode_calls_bm25(client):
    with patch("app.search.router.bm25_search", return_value=MOCK_RESULTS) as m:
        response = client.get("/search?q=hello&mode=bm25")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_hybrid_mode_calls_hybrid(client):
    with patch("app.search.router.hybrid_search", return_value=MOCK_HYBRID_RESULTS) as m:
        response = client.get("/search?q=hello&mode=hybrid")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_hybrid_rerank_mode_calls_hybrid_rerank(client):
    with patch("app.search.router.hybrid_rerank_search", return_value=MOCK_HYBRID_RESULTS) as m:
        response = client.get("/search?q=hello&mode=hybrid%2Brerank")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_invalid_mode_returns_422(client):
    response = client.get("/search?q=hello&mode=unknown")
    assert response.status_code == 422


def test_search_hybrid_rerank_no_key_returns_400(client):
    with patch("app.search.router.hybrid_rerank_search",
               side_effect=RuntimeError("COHERE_API_KEY not set")):
        response = client.get("/search?q=hello&mode=hybrid%2Brerank")
    assert response.status_code == 400
    assert "COHERE_API_KEY" in response.json()["detail"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_search_router.py -v
```
Expected: 6 new tests FAIL (router doesn't have `mode` yet).

- [ ] **Step 3: Update `backend/app/search/router.py`**

```python
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .bm25 import bm25_search
from .searcher import hybrid_rerank_search, hybrid_search, semantic_search

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_search_router.py -v
```
Expected: 10 passed (4 existing + 6 new).

- [ ] **Step 5: Run full test suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all tests pass (38 existing + new tests for Tasks 1-7).

- [ ] **Step 6: Commit**

```bash
git add backend/app/search/router.py backend/tests/test_search_router.py
git commit -m "feat(etap-2): add ?mode= parameter to GET /search (semantic/bm25/hybrid/hybrid+rerank)"
```

---

## Task 8: Search comparison script

**Files:**
- Create: `eval/__init__.py` (empty)
- Create: `eval/search_comparison.py`

CLI script — not unit tested. Requires a live DB with ingested audio and `OPENAI_API_KEY` (for semantic/hybrid modes). Runs all four modes on a given query and prints a markdown table. Use `.env` via `python-dotenv` or set env vars manually before running.

- [ ] **Step 1: Create `eval/__init__.py`**

Empty file.

- [ ] **Step 2: Create `eval/search_comparison.py`**

```python
#!/usr/bin/env python3
"""
Compare all four search modes on a single query.

Usage:
    cd <project-root>
    python -m eval.search_comparison "your search query" [--limit 5]

Requires:
    DATABASE_URL and OPENAI_API_KEY set in environment or .env file
    COHERE_API_KEY set for hybrid+rerank mode (optional; that mode is skipped otherwise)
"""
import argparse
import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.search.bm25 import bm25_search
from app.search.reranker import rerank
from app.search.rrf import reciprocal_rank_fusion
from app.search.searcher import hybrid_search, hybrid_rerank_search, semantic_search


def get_session():
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    engine = create_engine(url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session()


def fmt_result(i: int, chunk: dict) -> str:
    ts = f"{chunk['start_ts']:.1f}s–{chunk['end_ts']:.1f}s"
    score = f"{chunk.get('similarity', 0.0):.4f}"
    preview = chunk["text"][:80].replace("\n", " ")
    return f"{i+1}. [{score}] {ts} | {preview}"


def run_mode(name: str, fn, query: str, limit: int, db) -> list[str]:
    try:
        results = fn(query=query, limit=limit, db=db)
        return [fmt_result(i, r) for i, r in enumerate(results)]
    except Exception as e:
        return [f"ERROR: {e}"]


def main():
    parser = argparse.ArgumentParser(description="Compare search modes")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    db = get_session()
    query = args.query
    limit = args.limit

    modes = {
        "semantic":      lambda q, l, d: semantic_search(q, l, d),
        "bm25":          lambda q, l, d: bm25_search(q, l, d),
        "hybrid":        lambda q, l, d: hybrid_search(q, l, d),
        "hybrid+rerank": lambda q, l, d: hybrid_rerank_search(q, l, d),
    }

    lines = [f"# Search comparison: '{query}' (limit={limit})\n"]
    for mode_name, fn in modes.items():
        lines.append(f"## {mode_name}")
        results_lines = run_mode(mode_name, fn, query, limit, db)
        lines.extend(results_lines)
        lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "search_comparison_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {out_path}")

    db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run a quick import check**

```bash
cd backend && python -c "import sys; sys.path.insert(0, '.'); from app.search.bm25 import bm25_search; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add eval/__init__.py eval/search_comparison.py
git commit -m "feat(etap-2): add search comparison script for all four modes"
```

---

## Definition of Done (manual — requires real audio + API keys)

After all tasks complete, verify with a real ingested episode:

```bash
# Semantic
curl "localhost:8000/search?q=your+topic&mode=semantic&limit=3"

# BM25
curl "localhost:8000/search?q=your+topic&mode=bm25&limit=3"

# Hybrid (RRF)
curl "localhost:8000/search?q=your+topic&mode=hybrid&limit=3"

# Hybrid + Rerank (requires COHERE_API_KEY in .env)
curl "localhost:8000/search?q=your+topic&mode=hybrid%2Brerank&limit=3"

# Comparison script
python -m eval.search_comparison "your topic" --limit 5
```

All four endpoints return results. The comparison script saves `docs/search_comparison_results.md`. Record observations in `DECISIONS.md`: which mode returned most relevant results, what BM25 missed that semantic caught, etc.

---

## Self-Review

**Spec coverage:**
- [x] BM25 via `tsvector`/`ts_rank` → Task 3
- [x] RRF fusion (ranks not scores) → Task 4; `test_rrf_uses_ranks_not_raw_scores` test explicitly verifies
- [x] Reranking (Cohere Rerank) → Task 5
- [x] `?mode=semantic|bm25|hybrid|hybrid+rerank` on `GET /search` → Task 7
- [x] Comparison script generating output file → Task 8
- [x] Graceful degradation when `COHERE_API_KEY` unset → HTTP 400 (Task 7)

**Placeholder scan:** None found — every step has runnable code and exact commands.

**Type consistency:**
- `bm25_search` → `{..., "similarity": float}` (same shape as `semantic_search`)
- `reciprocal_rank_fusion` input: `list[list[dict]]` each dict has `"chunk_id"` key — produced by both `semantic_search` and `bm25_search` ✓
- `reciprocal_rank_fusion` output: adds `"rrf_score"` key — `hybrid_search` maps this to `"similarity"` before returning ✓
- `rerank` input: `list[dict]` with `"text"` key — all search result dicts have it ✓
- `hybrid_rerank_search` calls `hybrid_search` with `limit=limit*2` then `rerank` with `top_n=limit` — correct ✓
- `SearchResult.similarity` stays the score field across all modes ✓
