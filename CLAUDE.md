# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Voice Knowledge Agent — portfolio project for Voice AI Engineer specialization. Audio library (podcasts/lectures) → transcription → hybrid search → LangGraph agent → voice answers with timestamp citations. Full roadmap: `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md`.

## Stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy/psycopg, LangGraph, pytest
- **DB:** PostgreSQL 16 + pgvector (hybrid search: semantic + BM25 via `tsvector`)
- **STT/TTS:** OpenAI Whisper (API), ElevenLabs API (direct httpx, not SDK)
- **Frontend:** Next.js 15 (App Router), Tailwind CSS
- **MCP:** MCP Python SDK (`mcp>=1.0`) — separate venv required (starlette conflict)
- **Eval:** custom harness (`eval/`) — LLM-as-judge via Anthropic Haiku

## Running the stack

```bash
docker compose up                # starts postgres (5433) + backend (8000) + frontend (3000)
curl localhost:8000/health       # -> {"status": "ok"}
open http://localhost:3000       # UI
```

Backend hot-reloads via volume mount (`./backend:/app`).

## Running tests

```bash
cd backend
python -m pytest tests/ -v       # 244 tests, all mocked (no real DB/API calls needed)
```

## Git conventions

- **No `Co-Authored-By: Claude Code` in commits** — omit entirely.
- Commit format: `<type>(<scope>): <subject>` where type is `feat`, `fix`, `refactor`, `test`, `chore`, `docs`.
- Scope is the etap or module: `etap-1`, `ingest`, `search`, `agent`, `eval`, `mcp`, `frontend`.
- **One squash-commit per etap on `main`** — granular task commits live on branch; squash before merge with `git reset --soft $(git merge-base HEAD main)`.

## Architecture

```
backend/app/
  main.py          # FastAPI app + router mounts
  ingest/          # upload endpoint, Whisper transcription, chunker
  search/          # semantic + BM25 + RRF fusion + reranker (Cohere)
  agent/           # LangGraph graph + tools
  voice/           # STT/TTS endpoints
  episodes/        # list/get episodes and chunks

eval/
  scorer.py        # RAGTriadScorer (LLM-as-judge, 3 metrics)
  runner.py        # EvalRunner (calls backend HTTP)
  evaluate.py      # CLI: python -m eval.evaluate
  golden_dataset/  # 15 synthetic questions (simple/comparison/multi_hop)

mcp-server/        # Standalone MCP server (separate venv)
  tools.py         # httpx wrappers (testable in backend venv)
  server.py        # FastMCP registration

frontend/          # Next.js App Router
  app/             # pages
  components/      # AudioRecorder, ChatUI, SourcePlayer
```

## Critical data shape — chunk record (invariant across all etaps)

```
id UUID, episode_id UUID, start_ts FLOAT, end_ts FLOAT, text TEXT, embedding VECTOR(1536)
```

`start_ts`/`end_ts` are seconds from audio start. All timestamp citations depend on this. Never break this invariant.

## Search modes

`GET /search?q=...&mode=semantic|bm25|hybrid|hybrid+rerank&limit=N`

- `semantic`: pgvector cosine similarity (`<=>` operator), `WHERE embedding IS NOT NULL`
- `bm25`: PostgreSQL `tsvector` / `ts_rank`, GIN index on stored generated column
- `hybrid`: RRF(semantic, bm25) — fuses **ranks**, not raw scores, k=60
- `hybrid+rerank`: hybrid → Cohere Rerank API (requires `COHERE_API_KEY`; on `GET /search` this raises if unset — no fallback there)

## LangGraph agent tools

Defined in `backend/app/agent/tools.py` via `make_tools(db, mode="hybrid", sources_sink=None)` closure. `mode` selects which search function `search_transcripts`/`compare_across_episodes` use (Etap 14, selectable in the chat UI); `hybrid+rerank` silently falls back to `hybrid` if `COHERE_API_KEY` is unset (the agent has no clean way to surface a tool error to the user). `sources_sink`, if given, collects every retrieved chunk so `stream_agent_response` can emit them as an SSE `sources` event (deduplicated, capped at `MAX_SOURCES = 8`, projected to `{episode_id, start_ts, end_ts, text}`).

| Tool | Signature | What it does |
|------|-----------|-------------|
| `search_transcripts` | `(query: str, limit: int=5)` | search across all chunks via the selected mode (limit clamped to 1-20) |
| `get_context_around_timestamp` | `(episode_id: str, timestamp: float, window_seconds: float=30.0)` | chunks near a timestamp |
| `compare_across_episodes` | `(query: str, episode_ids: str, limit_per_episode: int=3)` | grouped search per episode (limit clamped to 1-20) |
| `summarise_segment` | `(text: str)` | format segment for LLM analysis |

MAX_STEPS = 6 (step counter guard in graph state). MemorySaver checkpoints per `thread_id` (module-level singleton — see gotchas).

**Chat history** (`backend/app/agent/history.py`, `db/models.py::ChatThread`/`ChatMessage`): every `/agent/chat` and `/agent/chat/stream` call persists the user + assistant messages (with citations) to Postgres, separate from LangGraph's own checkpointing. `GET /agent/threads` lists past conversations (id, title, updated_at); `GET /agent/threads/{id}` returns the full message list for reopening one. Thread title is the first ~60 chars of the first user message.

## MCP tools

Defined in `mcp-server/tools.py` — pure httpx, no MCP dependency, testable in backend venv.

| Tool | Returns |
|------|---------|
| `search_knowledge_base(query, limit=5)` | formatted chunk list with timestamps |
| `get_episode_summary(episode_id)` | episode metadata + first 10 transcript chunks |
| `find_mentions(topic, limit=10)` | chunk list where topic is mentioned |

Chunks returned as `{chunk + timestamp + episode}` — never full transcripts (context window hygiene).

## Eval harness

```bash
# Requires running backend + ANTHROPIC_API_KEY
python -m eval.evaluate \
  --base-url http://localhost:8000 \
  --modes semantic bm25 hybrid hybrid+rerank \
  --output docs/eval_report \
  --limit 5    # optional quick smoke test
```

RAG triad: Context Relevance, Groundedness, Answer Relevance — each 0.0–1.0.
LLM-as-judge: Anthropic Haiku, 1-5 integer scale, normalized to (val-1)/4.0.

## Key mocking patterns for tests

```python
# OpenAI lazy clients (lru_cache)
with patch("app.ingest.embedder._client", return_value=mock_openai):
    ...

# ElevenLabs TTS (httpx directly, no SDK)
with patch("app.voice.tts.httpx") as mock_httpx:
    mock_httpx.post.return_value = ...

# Cohere reranker (lru_cache client)
with patch("app.search.reranker._client", return_value=mock_cohere):
    ...

# FastAPI dependency injection (DB session)
app.dependency_overrides[get_db] = lambda: mock_db
```

## Environment variables

Copy `.env.example` → `.env` (never commit `.env`).

| Variable | Required for |
|---|---|
| `DATABASE_URL` | All etaps (local: `postgresql://voicerag:voicerag@localhost:5433/voicerag`) |
| `OPENAI_API_KEY` | Ingest (Whisper + embeddings), Voice STT |
| `ANTHROPIC_API_KEY` | Agent (LangGraph LLM), Eval (LLM-as-judge) |
| `ELEVENLABS_API_KEY` | Voice TTS |
| `COHERE_API_KEY` | hybrid+rerank search mode (optional — falls back to hybrid) |
| `ALLOWED_ORIGINS` | CORS (Etap 13) — comma-separated frontend origins, default `http://localhost:3000` |
| `RATE_LIMIT_ENABLED` | Rate limiting (Etap 13) — `"false"` disables it (tests default it off); default `true` |
| `API_KEY` | Optional `X-API-Key` gate on `/ingest/*` (Etap 13) — blank disables the gate; `frontend/middleware.ts` injects it server-side when set |

## Known gotchas

- **ElevenLabs SDK**: do NOT install `elevenlabs>=1` on Windows — path lengths >260 chars cause `OSError`. Use `httpx.post` directly to the REST API (already implemented).
- **MCP venv conflict**: `mcp>=1.0` requires `starlette>=1.0.0`; `fastapi 0.115.0` requires `starlette<0.39.0`. Keep `mcp-server/` in a separate venv.
- **LangGraph import**: `from langgraph.graph.message import add_messages` (not `langchain_core.messages`).
- **pgvector cosine**: use `<=>` operator for cosine distance; always add `WHERE embedding IS NOT NULL` guard.
- **RRF**: sum reciprocal **ranks** (1/(k+rank)), not raw similarity scores. Verify with test that puts a chunk in both lists — it should rank first in fused results.
- **Whisper STT for voice queries**: returns plain `str`, not segment list. Only ingest transcriber returns segments (for timestamp chunking).
- **Test suite**: 244 tests, all mocked. Never require a running database or real API keys for tests.
- **Anthropic streaming chunk shape**: `chunk.content` from `ChatAnthropic().astream_events()` (langchain-anthropic 1.x) is a **list of content blocks** (`[{"type": "text", "text": "..."}]`, interleaved with `tool_use`/`input_json_delta` blocks during a tool call), not a plain string. `app.agent.streaming._extract_text` handles this; don't reintroduce a bare `isinstance(content, str)` check — it silently drops every token (this shipped broken once already; mocked tests didn't catch it because the mock used a plain string).
- **`:param::type` in raw SQL**: SQLAlchemy's `text()` bind-param regex has a negative lookahead on a trailing `:`, so `:emb::vector` is never recognized as a bind param (collides with Postgres's `::` cast) — it's silently left as literal text and the param is dropped before reaching the driver. Use `CAST(:param AS type)` instead. Same root cause as above: mocked-DB tests never compile the SQL against a real dialect, so this also shipped broken once.
- **`MemorySaver` must be a module-level singleton**: `build_graph()` is called fresh on every `/agent/chat` and `/agent/chat/stream` request. A `MemorySaver()` instantiated *inside* `build_graph()` gives every request an empty checkpoint store — `thread_id` becomes inert and the agent has zero memory of earlier turns, even within one browser session (this shipped broken once too — no test called `build_graph()` twice with the same `thread_id` to catch it). The checkpointer (`app.agent.graph._CHECKPOINTER`) must be created once at module scope and reused across calls.
- **Chat history persistence must happen before TTS, not after**: in `stream_agent_response`, save the assistant message (`app.agent.history.save_message`) right after the token loop ends — *before* the TTS `await` loop. A TTS failure is fatal-by-design (aborts before `"done"`), so persistence placed after it silently never runs whenever ElevenLabs is unavailable (this shipped broken once with `ELEVENLABS_API_KEY` unset — every streamed answer's history entry was lost).

## Ingest commands

```bash
# Upload local file
curl -X POST localhost:8000/ingest/upload -F "file=@recording.mp3"
# Returns: {"id": "<uuid>", "filename": "...", "chunk_count": N}

# Ingest from URL
curl -X POST localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/podcast.mp3"}'
```

## Documentation conventions

- **`DECISIONS.md`** — every non-trivial technical decision: chosen, rejected alternatives, why. Written at end of each etap.
- **`docs/claude-md-fragments/etap-N.md`** — per-etap detail that fed into this final `CLAUDE.md`.
- Plans: `docs/superpowers/plans/YYYY-MM-DD-etap-N-<name>.md`
