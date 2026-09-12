# Voice Knowledge Agent

[![CI](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml)

Ask questions about your podcast/lecture library — by voice or text — and get spoken answers with timestamp citations you can click to jump to the exact moment in the source recording.

📘 **[Project walkthrough (interactive)](https://claude.ai/code/artifact/36785123-1034-4248-9c00-0b886d62faf9)** — the architecture below, step by step, with mock data at every stage, why LangGraph exists, the full MCP-vs-agent split, and a with/without-LangGraph comparison. Private artifact — open it signed into the account that created it.

🔗 **[Live demo](https://voice-rag-agent-ten.vercel.app)** — frontend on Vercel, backend + Postgres/pgvector on Railway. It's a low-traffic portfolio deployment on free/low-cost hosting tiers (see `DEPLOYMENT.md`), not sized for sustained heavy traffic.

---

## Architecture

```
Browser (Next.js 15)
  │  voice input → MediaRecorder (energy-based VAD auto-stop) → webm blob
  │  text input  → fetch
  │  search mode → semantic | bm25 | hybrid | hybrid+rerank selector
  ▼
FastAPI backend (port 8000)
  ├── POST /voice/stt          → OpenAI Whisper → transcript
  ├── POST /agent/chat         → LangGraph agent (single response)
  ├── POST /agent/chat/stream  → LangGraph agent (SSE: tokens, sources, done)
  │     ├── search_transcripts      → hybrid search (RRF + optional Cohere rerank)
  │     ├── get_context_around_ts   → timestamp lookup
  │     ├── compare_across_episodes → per-episode grouped search
  │     └── summarise_segment       → segment formatting
  ├── POST /voice/tts          → ElevenLabs → MP3 bytes, streamed sentence-by-sentence
  ├── POST /ingest/upload | /ingest/url | /ingest/rss   → Whisper transcription + chunking + embedding
  └── GET  /search?q=...&mode= → direct hybrid/semantic/bm25/rerank search
                                           │
                              ◄────────────┘
                          audio plays in browser (AudioContext streaming playback)
                          source citations appear with ▶ jump links (Media Fragments seek)

PostgreSQL 16 + pgvector
  ├── episodes table (id, filename, source_url)
  └── chunks table (id, episode_id, start_ts, end_ts, text, embedding vector(1536))
       ├── GIN index on tsvector column (BM25)
       └── ivfflat/hnsw index on embedding (semantic)

MCP Server (stdio)
  ├── search_knowledge_base(query)
  ├── get_episode_summary(episode_id)
  └── find_mentions(topic)
```

Security hardening (SSRF guard on `/ingest/url`+`/ingest/rss`, upload content-sniffing, rate limiting, optional `X-API-Key` gate on ingest, CORS) is documented in `DECISIONS.md`.

---

## Evaluation

Eval harness: `eval/evaluate.py` runs RAG triad (Context Relevance, Groundedness, Answer Relevance) across all retrieval modes using an LLM-as-judge (Claude Haiku).

To run:
```bash
export ANTHROPIC_API_KEY=sk-...
# backend must be running at localhost:8000 with ingested content
python -m eval.evaluate --modes semantic bm25 hybrid hybrid+rerank --output docs/eval_report
```

Results are written to `docs/eval_report.md` (markdown table) and `docs/eval_report.json` (raw scores).

**Measured** on one real episode of NASA's "Houston We Have a Podcast" (episode 435,
"National Lab 15" — a ~24min interview about the ISS National Lab program), 8 real
questions written against the actual transcript, RAG triad scored by Claude Haiku
(2026-09-08):

| Mode | Context Relevance | Groundedness | Answer Relevance | Composite |
|------|:-----------------:|:------------:|:----------------:|:---------:|
| semantic | 0.812 | 0.906 | 0.938 | 0.885 |
| bm25 | 0.031 | 0.031 | 0.969 | 0.344 |
| hybrid | 0.812 | 0.938 | 0.938 | **0.896** |
| hybrid+rerank | — | — | — | not yet measured (needs `COHERE_API_KEY`) |

**Why BM25 collapses here**: `bm25_search` (`backend/app/search/bm25.py`) builds its
query with `plainto_tsquery`, which ANDs together every stemmed word in the question.
These are 10-20 word natural-language questions ("What challenge does the ISS National
Lab face when working with commercial partners...") — requiring every one of those
stems to co-occur in a single ~60-100 word chunk returns **zero rows** for most of them
(`q01`, `q02`, `q04`-`q08` all retrieved 0 chunks). Answer Relevance stays high anyway
because `/agent/chat` always searches in `hybrid` mode internally regardless of what
mode the eval harness asked `/search` for — so the *answer* is still well-grounded, only
the context fed to the judge for `bm25` is the (empty) BM25-only result. This is exactly
the paraphrase-query failure mode hybrid search exists to fix (see `DECISIONS.md`,
Etap 2) — now with real numbers instead of a prediction. Full per-question breakdown:
[`docs/eval_report.md`](docs/eval_report.md).

Run `python -m eval.evaluate` after ingesting more content to extend this table.

---

## MCP Server

Exposes the knowledge base as MCP tools for Claude Desktop / Claude Code:

```bash
cd mcp-server
pip install -r requirements.txt  # separate venv required (starlette version conflict)
python server.py
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "voice-knowledge": {
      "command": "python",
      "args": ["/absolute/path/to/Voice-RAG-Agent/mcp-server/server.py"],
      "env": {"BACKEND_URL": "http://localhost:8000"}
    }
  }
}
```

Tools: `search_knowledge_base`, `get_episode_summary`, `find_mentions`

---

## Local Setup

```bash
# 1. Clone and copy env
git clone <repo-url>
cd Voice-RAG-Agent
cp .env.example .env
# Edit .env — add OPENAI_API_KEY, ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, COHERE_API_KEY

# 2. Start the stack
docker compose up          # postgres (5433) + backend (8000) + frontend (3000)

# 3. Verify
curl localhost:8000/health  # {"status": "ok"}
open http://localhost:3000  # UI

# 4. Ingest an audio file
curl -X POST localhost:8000/ingest/upload \
  -F "file=@my-podcast.mp3"

# 5. Ask a question (text)
curl -X POST localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main argument about AI replacing knowledge workers?", "thread_id": "test-1"}'
```

---

## Technical Decisions

Key choices documented in [`DECISIONS.md`](DECISIONS.md):

- **Hybrid search**: pgvector cosine similarity + PostgreSQL `tsvector` BM25 fused with Reciprocal Rank Fusion (k=60). Pure semantic misses exact-match queries; pure BM25 misses paraphrase queries; RRF reliably beats both.
- **Reranker**: Cohere Rerank API over local cross-encoder — GPU not available in dev; same interface, easy to swap.
- **LangGraph agent**: `StateGraph` with step counter guard (MAX_STEPS=6) and `MemorySaver` checkpointing — genuinely routes differently for simple vs. multi-hop queries.
- **ElevenLabs TTS**: raw `httpx.post` instead of SDK — SDK fails on Windows with path lengths >260 chars.
- **MCP server venv**: `mcp>=1.0` conflicts with `fastapi 0.115` on starlette — kept in separate venv, tools.py has no MCP dependency so tests run in the backend venv.
- **Eval LLM-as-judge**: Haiku instead of Sonnet/Opus — 15q × 4 modes × 3 metrics = 180 API calls; Haiku is sufficient for 1-5 integer scoring.

---

## Known Limitations

- **Low-traffic demo hosting**: the [live demo](https://voice-rag-agent-ten.vercel.app) runs on Railway's lowest paid tier + Vercel's free Hobby tier (see `DEPLOYMENT.md` for cost notes and deploy steps) — fine for a portfolio link, not sized for sustained traffic. Local `docker compose up` remains the way to run the full stack for development.
- **compare_across_episodes**: the LangGraph tool exists but is not deeply tested for cross-episode queries requiring multiple tool calls against specific episode IDs.
- **Single-episode golden dataset**: the eval corpus is currently one real episode (kept small deliberately — Whisper transcription is metered API spend). The 8 questions above are real and grounded in that transcript, but don't yet exercise `compare_across_episodes`; ingest more episodes and extend `eval/golden_dataset/questions.json` to cover that.
- **No persistent audio storage for uploads**: `SourcePlayer` can only seek into episodes ingested via URL/RSS (which keep an external `source_url`); directly-uploaded files are transcribed and discarded, so their citations show text only, no jump-to-timestamp playback.

---

## Project Structure

```
backend/           FastAPI app (Python 3.12)
  app/
    ingest/        upload, Whisper transcription, chunking, embedding, storage
    search/        semantic, BM25, hybrid (RRF), reranker (Cohere)
    agent/         LangGraph graph, tools, chat endpoint
    voice/         STT (Whisper), TTS (ElevenLabs)
    episodes/      list/get episodes and their chunks
    db/            SQLAlchemy models, session, init_db
  tests/           225 pytest tests, all mocked (no real DB/API calls)

eval/              RAG eval harness
  scorer.py        RAGTriadScorer (LLM-as-judge)
  runner.py        EvalRunner (calls backend HTTP)
  evaluate.py      CLI entry point
  golden_dataset/  15 synthetic Q&A pairs

mcp-server/        Standalone MCP server
  tools.py         httpx wrappers for backend endpoints
  server.py        FastMCP registration

frontend/          Next.js 15 App Router
  components/
    AudioRecorder  MediaRecorder → POST /voice/stt
    ChatUI         text+voice input, TTS playback
    SourcePlayer   chunk citations with timestamp display
```
