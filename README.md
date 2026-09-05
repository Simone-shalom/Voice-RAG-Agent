# Voice Knowledge Agent

Ask questions about your podcast/lecture library — by voice or text — and get spoken answers with timestamp citations you can click to jump to the exact moment in the source recording.

---

## Architecture

```
Browser (Next.js 15)
  │  voice input → MediaRecorder → webm blob
  │  text input → fetch
  ▼
FastAPI backend (port 8000)
  ├── POST /voice/stt   → OpenAI Whisper → transcript
  ├── POST /agent/chat  → LangGraph agent
  │     ├── search_transcripts      → hybrid search (RRF + optional Cohere rerank)
  │     ├── get_context_around_ts   → timestamp lookup
  │     ├── compare_across_episodes → per-episode grouped search
  │     └── summarise_segment       → segment formatting
  └── POST /voice/tts   → ElevenLabs → MP3 bytes
                                           │
                              ◄────────────┘
                          audio plays in browser
                          source citations appear with ▶ jump links

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

**Expected pattern** (based on RAG literature; measure on your own corpus):
| Mode | Context Relevance | Groundedness | Answer Relevance | Composite |
|------|:-----------------:|:------------:|:----------------:|:---------:|
| semantic | — | — | — | — |
| bm25 | — | — | — | — |
| hybrid | — | — | — | — |
| hybrid+rerank | — | — | — | — |

Run `python -m eval.evaluate` after ingesting content to populate this table.

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

- **No streaming**: voice answers play after full generation. Streaming STT→LLM→TTS would require WebSocket / SSE and significantly lower the time-to-first-sound.
- **No VAD**: silence detection (Voice Activity Detection) is not implemented; recording stops on button click, not automatically.
- **Synthetic golden dataset**: `eval/golden_dataset/questions.json` contains 15 questions written for a hypothetical corpus. Eval scores are only meaningful after ingesting real content and updating the questions.
- **No production deploy**: Docker Compose runs everything locally. Vercel (frontend) + Railway (backend+postgres) deployment is left as an exercise — all required env vars are documented.
- **compare_across_episodes**: the LangGraph tool exists but is not deeply tested for cross-episode queries requiring multiple tool calls against specific episode IDs.

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
  tests/           131 pytest tests, all mocked (no real DB/API calls)

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
