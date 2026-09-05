# Technical Specification — Voice Knowledge Agent

> Portfolio project: Voice AI Engineer specialization. Last updated: 2026-09-05.

---

## 1. System Overview

Voice Knowledge Agent is a multi-modal retrieval-augmented generation (RAG) system that lets users query an audio content library by voice or text and receive spoken answers with transcript-level timestamp citations.

**Primary flows:**
1. **Ingest** — upload audio → Whisper transcription → chunking → embedding → store in PostgreSQL/pgvector
2. **Query** — text/voice input → LangGraph agent → hybrid retrieval → LLM synthesis → TTS output + source citations
3. **Evaluate** — automated RAG triad scoring across all retrieval modes using LLM-as-judge

**Services (Docker Compose):**

| Service   | Port | Technology |
|-----------|------|------------|
| postgres  | 5433 | PostgreSQL 16 + pgvector extension |
| backend   | 8000 | Python 3.12, FastAPI 0.115, SQLAlchemy/psycopg |
| frontend  | 3000 | Next.js 15 App Router, Tailwind CSS |

---

## 2. Data Pipeline (Ingest → Chunk → Embed → Store)

### 2.1 Input

`POST /ingest/upload` accepts `multipart/form-data` with an audio file (MP3, WAV, M4A, WebM). `POST /ingest/url` accepts a JSON body with a `url` field and downloads the file before processing.

### 2.2 Transcription

OpenAI Whisper API (`whisper-1`). For ingest, the raw response includes word-level segments with `start`/`end` timestamps. The STT voice endpoint uses Whisper for live queries but discards segment data — it returns a plain string.

```python
# backend/app/ingest/transcriber.py
response = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json"
)
# response.segments[i] → {text, start, end}
```

### 2.3 Chunking

`backend/app/ingest/chunker.py` groups Whisper segments into ~30-second windows with a 5-second overlap between consecutive chunks.

**Chunk record invariant** (never break this — all citation features depend on it):

```sql
chunks (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  episode_id UUID REFERENCES episodes(id),
  start_ts   FLOAT NOT NULL,   -- seconds from audio start
  end_ts     FLOAT NOT NULL,   -- seconds from audio start
  text       TEXT NOT NULL,
  embedding  VECTOR(1536)      -- OpenAI text-embedding-ada-002
)
```

### 2.4 Embedding

`backend/app/ingest/embedder.py` calls OpenAI `text-embedding-ada-002` (1536 dimensions). The client is instantiated lazily behind `@lru_cache` to avoid import-time API key requirements.

### 2.5 Storage

SQLAlchemy/psycopg (sync) writes to PostgreSQL. Two indexes on the `chunks` table:

```sql
-- semantic search (pgvector)
CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);

-- BM25 search
ALTER TABLE chunks ADD COLUMN ts_text TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX chunks_ts_text_idx ON chunks USING GIN (ts_text);
```

---

## 3. Search Subsystem

`GET /search?q=...&mode=semantic|bm25|hybrid|hybrid+rerank&limit=N`

### 3.1 Semantic Search

```sql
SELECT *, 1 - (embedding <=> :query_embedding) AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_embedding
LIMIT :limit;
```

The `<=>` operator is pgvector cosine distance (lower = more similar); `1 - distance` converts to similarity. The `WHERE embedding IS NOT NULL` guard prevents null-vector rows from silently poisoning rankings.

### 3.2 BM25 Search

```sql
SELECT *, ts_rank(ts_text, plainto_tsquery('english', :query)) AS score
FROM chunks
WHERE ts_text @@ plainto_tsquery('english', :query)
ORDER BY score DESC
LIMIT :limit;
```

Uses PostgreSQL's native full-text search. `ts_rank` is not a true BM25 (no IDF saturation), but behaves comparably for this use case and requires no additional extension.

### 3.3 Hybrid Search — Reciprocal Rank Fusion

RRF combines the two ranked lists without requiring score normalization. For a chunk appearing at rank `r` in a list, its contribution is `1 / (k + r)` where `k=60` (standard literature default).

```python
def rrf_fuse(semantic_ids, bm25_ids, k=60):
    scores = defaultdict(float)
    for rank, cid in enumerate(semantic_ids, 1):
        scores[cid] += 1 / (k + rank)
    for rank, cid in enumerate(bm25_ids, 1):
        scores[cid] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**Critical:** RRF operates on **ranks**, not raw scores. A chunk that appears at rank 1 in both lists scores `2/(k+1)`, beating any chunk that only appears in one list regardless of how high its raw similarity was. This is tested explicitly — a chunk seeded into both result sets must rank first in the fused output.

### 3.4 Reranking (hybrid+rerank)

After hybrid retrieval, the top-N chunks are sent to the Cohere Rerank API:

```python
# backend/app/search/reranker.py
response = client.rerank(
    model="rerank-english-v2.0",
    query=query,
    documents=[c.text for c in chunks],
    top_n=limit
)
```

The returned ordering replaces the RRF ordering. Cohere is a cross-encoder that reads each (query, document) pair jointly — it captures relevance signals that bi-encoder retrieval misses (e.g., negation, multi-part questions). Falls back to `hybrid` if `COHERE_API_KEY` is not set.

---

## 4. Agent Subsystem

### 4.1 LangGraph State Machine

```python
# backend/app/agent/graph.py
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages  # NOT langchain_core.messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int    # guard against infinite loops

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", route)  # → "tools" | END
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

checkpointer = MemorySaver()  # in-memory per thread_id
app = graph.compile(checkpointer=checkpointer)
```

`MAX_STEPS = 6`: the `steps` counter increments on every tool call; `route()` returns `END` when `steps >= MAX_STEPS` to prevent runaway loops.

### 4.2 Tool Definitions

All four tools are defined in `backend/app/agent/tools.py` via a `make_tools(db)` closure — the database session is captured in closure scope rather than injected per-call.

| Tool | Key behavior |
|------|-------------|
| `search_transcripts(query, limit=5)` | Calls internal hybrid search (same SQL path as `/search` endpoint), returns ranked chunks |
| `get_context_around_timestamp(episode_id, timestamp, window_seconds=30.0)` | `WHERE start_ts BETWEEN ts-window AND ts+window ORDER BY start_ts` |
| `compare_across_episodes(query, episode_ids, limit_per_episode=3)` | Runs semantic search filtered to each episode ID separately, groups results |
| `summarise_segment(text)` | No DB call — formats text as a structured prompt snippet for LLM analysis |

### 4.3 Thread Memory

`MemorySaver` stores `AgentState` snapshots keyed by `thread_id`. Each `POST /agent/chat` request carries a `thread_id`; follow-up questions in the same thread see prior messages. State is in-process memory — it resets on backend restart. For production use, swap `MemorySaver` for `PostgresSaver`.

---

## 5. Voice Layer

### 5.1 STT — Whisper

`POST /voice/stt` accepts `multipart/form-data` with an `audio` file (webm from MediaRecorder).

```python
# backend/app/voice/stt.py
def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as f:
        f.write(file_bytes); f.flush()
        with open(f.name, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
    return transcript.text  # plain str, no segments
```

Returns a plain string. Unlike ingest transcription, voice queries do not need segment timestamps.

### 5.2 TTS — ElevenLabs

`POST /voice/tts` accepts `{"text": "..."}` and returns `audio/mpeg` bytes.

```python
# backend/app/voice/tts.py
def synthesise(text: str) -> bytes:
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": api_key},
        json={"text": text, "model_id": "eleven_monolingual_v1"}
    )
    resp.raise_for_status()
    return resp.content
```

**Why not the ElevenLabs SDK?** The SDK (≥1.0) installs packages with path lengths exceeding 260 characters, causing `OSError` on Windows. The raw `httpx.post` approach avoids this entirely. See `DECISIONS.md` for rejected alternatives.

---

## 6. Evaluation Framework

### 6.1 RAG Triad Metrics

`eval/scorer.py` implements three metrics, each on a 0.0–1.0 scale:

| Metric | Question answered | How scored |
|--------|-------------------|------------|
| Context Relevance | Does the retrieved context actually answer the question? | LLM rates 1–5 |
| Groundedness | Is the answer supported by the context (no hallucination)? | LLM rates 1–5 |
| Answer Relevance | Does the answer address the original question? | LLM rates 1–5 |

Score normalization: `(raw_score - 1) / 4.0` maps the 1–5 integer range to [0.0, 1.0].

### 6.2 LLM-as-Judge

```python
# eval/scorer.py
def make_anthropic_judge(api_key, model="claude-haiku-4-5-20251001"):
    client = anthropic.Anthropic(api_key=api_key)
    def judge(prompt: str) -> str:
        return client.messages.create(
            model=model, max_tokens=16,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text
    return judge
```

Haiku is chosen over Sonnet/Opus: a full eval run (15 questions × 4 modes × 3 metrics) makes 180 API calls. Haiku handles 1-5 integer scoring at a fraction of the cost. `max_tokens=16` is tight enough to force a single digit response, parsed with `_parse_score()` which scans left-to-right for the first digit in [1,5].

### 6.3 EvalRunner

`eval/runner.py.EvalRunner.run_question()` calls `GET /search` (to capture retrieved chunks) then `POST /agent/chat` (to get the synthesized answer), then scores both together with `RAGTriadScorer.score_all()`. Errors per (question, mode) pair are captured individually — a network error on one pair doesn't abort the whole run.

### 6.4 Golden Dataset

`eval/golden_dataset/questions.json` — 15 synthetic questions calibrated to a hypothetical corpus:
- 5 simple (direct factual recall)
- 5 comparison (cross-episode, requires `compare_across_episodes` tool)
- 5 multi-hop (requires ≥2 tool calls to answer)

**Important:** these questions are written for a hypothetical library. Scores are only meaningful after ingesting a real corpus and updating the questions to match it.

---

## 7. MCP Integration

`mcp-server/` is a standalone FastMCP server that wraps the backend REST API. It runs in a **separate venv** because `mcp>=1.0` requires `starlette>=1.0.0`, which conflicts with `fastapi 0.115.0`'s requirement of `starlette<0.39.0`.

### 7.1 Tool implementations (`tools.py`)

Pure `httpx` — no `mcp` import — so tests run in the backend venv:

```python
def search_knowledge_base(query, base_url, limit=5, mode="hybrid") -> list[dict]:
    resp = httpx.get(f"{base_url}/search", params={"q": query, "mode": mode, "limit": limit})
    resp.raise_for_status(); return resp.json()

def format_chunks_for_mcp(chunks) -> str:
    if not chunks: return "No results found."
    return "\n\n".join(
        f"[{i}] Episode {c['episode_id']} @ {c['start_ts']:.1f}s–{c['end_ts']:.1f}s\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
```

### 7.2 FastMCP registration (`server.py`)

```python
mcp = FastMCP("Voice Knowledge Agent")

@mcp.tool()
def search_knowledge_base_tool(query: str, limit: int = 5) -> str:
    """Search podcast transcripts using hybrid retrieval."""
    return format_chunks_for_mcp(search_knowledge_base(query, BASE_URL, limit))
```

---

## 8. API Reference

### Ingest

| Method | Path | Body / Params | Response |
|--------|------|---------------|----------|
| POST | `/ingest/upload` | `multipart/form-data`: `file` | `{id, filename, chunk_count}` |
| POST | `/ingest/url` | `{"url": "..."}` | `{id, filename, chunk_count}` |

### Episodes

| Method | Path | Params | Response |
|--------|------|--------|----------|
| GET | `/episodes` | — | `[{id, filename, source_url, chunk_count}]` |
| GET | `/episodes/{id}` | — | `{id, filename, source_url, chunk_count}` or 404 |
| GET | `/episodes/{id}/chunks` | `?limit=20` | `[{id, start_ts, end_ts, text}]` |

### Search

| Method | Path | Params | Response |
|--------|------|--------|----------|
| GET | `/search` | `q`, `mode`, `limit` | `[{episode_id, start_ts, end_ts, text, score}]` |

### Agent

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | `/agent/chat` | `{"message": "...", "thread_id": "..."}` | `{"reply": "...", "sources": [...]}` |

### Voice

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | `/voice/stt` | `multipart/form-data`: `audio` | `{"text": "..."}` |
| POST | `/voice/tts` | `{"text": "..."}` | `audio/mpeg` bytes |

---

## 9. Performance Characteristics and Known Limits

| Characteristic | Value | Notes |
|----------------|-------|-------|
| Ingest latency | 3–6 min / hour of audio | Whisper API call; synchronous |
| Search latency (semantic) | ~50–200 ms | ivfflat index; scales with chunk count |
| Search latency (hybrid) | ~80–300 ms | Two SQL queries + in-process RRF fusion |
| Search latency (hybrid+rerank) | ~300–800 ms | Adds Cohere API round-trip |
| Agent response latency | 2–8 s | Depends on number of tool calls (max 6) |
| TTS latency | 500 ms–2 s | ElevenLabs API; full audio before playback |
| Eval full run | ~10–20 min | 180 Haiku API calls; network-bound |
| Max chunk size | ~30 s / ~100 words | Configured in `chunker.py` |
| Embedding dimension | 1536 | text-embedding-ada-002; pgvector VECTOR(1536) |

**Known limits:**
- No streaming audio responses — TTS completes before playback starts
- No VAD — users manually stop recording
- `MemorySaver` is in-process — conversation history lost on restart
- `compare_across_episodes` requires the caller to know episode IDs upfront
- No production deployment config (Docker Compose is local-only)

---

## 10. Extension Points

| Extension | What to change |
|-----------|----------------|
| Swap embedding model | `backend/app/ingest/embedder.py` + update `VECTOR(1536)` → new dim in SQL |
| Add streaming TTS | Replace `POST /voice/tts` with a chunked streaming endpoint; update `ChatUI.tsx` to consume `ReadableStream` |
| Persistent agent memory | Swap `MemorySaver` for `langgraph.checkpoint.postgres.PostgresSaver` |
| Production deploy | Vercel (frontend) + Railway (backend + postgres) — all env vars are documented |
| Custom voice | Change `VOICE_ID` env var to any ElevenLabs voice ID |
| Larger corpus index | Switch ivfflat → HNSW for better recall at scale (`CREATE INDEX ... USING hnsw`) |
