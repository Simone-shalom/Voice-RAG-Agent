# CLAUDE.md fragment — Etap 2: Hybrid Search

## Search modes

`GET /search?q=<query>&mode=<mode>&limit=<N>` — `mode` defaults to `semantic`.

| Mode | Description |
|------|-------------|
| `semantic` | pgvector cosine similarity (Etap 1) |
| `bm25` | PostgreSQL `tsvector`/`ts_rank` full-text search |
| `hybrid` | RRF fusion of semantic + BM25 (2× fetch headroom, k=60) |
| `hybrid+rerank` | hybrid + Cohere Rerank v3; requires `COHERE_API_KEY` |

## Key modules added

| Module | Purpose |
|--------|---------|
| `backend/app/search/bm25.py` | BM25 via `plainto_tsquery` + `ts_rank`, GIN-indexed `text_search` column |
| `backend/app/search/rrf.py` | Pure `reciprocal_rank_fusion(result_lists, k=60)` — fuses ranks, not scores |
| `backend/app/search/reranker.py` | Cohere Rerank lazy client; `RuntimeError` if key unset |
| `backend/app/search/searcher.py` | Extended with `hybrid_search`, `hybrid_rerank_search` |
| `eval/search_comparison.py` | CLI: `python -m eval.search_comparison "query" [--limit N]` |

## DB migration (idempotent, runs at startup)

`init_db` in `session.py` now runs:
1. `CREATE EXTENSION IF NOT EXISTS vector`
2. `DO $$ BEGIN IF NOT EXISTS ... ADD COLUMN text_search tsvector GENERATED ALWAYS AS ... END $$`
3. `CREATE INDEX IF NOT EXISTS chunks_text_search_gin ON chunks USING GIN(text_search)`

## RRF invariant

RRF fuses **ranks** (ordinal position 1, 2, 3...), not raw cosine or ts_rank scores. Score `1/(60+rank)` per list, summed per chunk. Never use raw similarity values as fusion input — they're on incompatible scales.

## Error handling

`hybrid+rerank` without `COHERE_API_KEY` → `RuntimeError("COHERE_API_KEY not set")` → HTTP 400. No silent fallback — caller must handle explicitly.

## Test mocking pattern

```python
# RRF (pure function — no mocking needed)
from app.search.rrf import reciprocal_rank_fusion

# Reranker lazy client
with patch("app.search.reranker._client", return_value=mock_cohere_client):
    result = rerank("query", chunks, top_n=3)

# Hybrid dispatch in router
with patch("app.search.router.hybrid_search", return_value=mock_results):
    response = client.get("/search?q=hello&mode=hybrid")
```

## Decisions made (see DECISIONS.md for full rationale)

- PostgreSQL `tsvector` over Elasticsearch (no new service; good enough BM25 approximation for same-length audio chunks)
- RRF over score normalization (no calibration needed; k=60 standard default)
- Cohere Rerank over local cross-encoder (no GPU on dev; same interface for easy swap)
