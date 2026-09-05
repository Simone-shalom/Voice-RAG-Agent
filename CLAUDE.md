# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Voice Knowledge Agent — portfolio project for Voice AI Engineer specialization. Audio library (podcasts/lectures) → transcription → hybrid search → LangGraph agent → voice answers with timestamp citations. Full roadmap: `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md`.

## Stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy/psycopg, LangGraph, pytest
- **DB:** PostgreSQL 16 + pgvector (hybrid search: semantic + BM25 via `tsvector`)
- **STT/TTS:** OpenAI Whisper (or `faster-whisper`), ElevenLabs API
- **Frontend:** Next.js 15 (App Router), Vercel AI SDK — added in Etap 4
- **MCP:** MCP Python SDK — added in Etap 6
- **Eval:** custom harness (`eval/`) — added in Etap 5

## Running the stack

```bash
docker compose up                # starts postgres (port 5433) + backend (port 8000)
curl localhost:8000/health       # -> {"status": "ok"}
```

Backend hot-reloads via volume mount (`./backend:/app`). Postgres persists data in `pgdata` volume.

## Git conventions

- **No `Co-Authored-By: Claude Code` in commits** — omit entirely.
- Commit format: `<type>(<scope>): <subject>` where type is `feat`, `fix`, `refactor`, `test`, `chore`, `docs`.
- Scope is the etap or module: `etap-1`, `ingest`, `search`, `agent`, `eval`, `mcp`, `frontend`.
- Subject in English, imperative mood, no period.
- Example: `feat(etap-1): implement ingest pipeline with Whisper transcription and pgvector search`
- **One squash-commit per etap on `main`** — granular task commits live on the branch during development and review; before merge, squash the entire branch into a single well-written commit summarising all changes. Use `git reset --soft $(git merge-base HEAD main)` then commit.
- After squash, force-push the branch with `git push --force-with-lease` before merging.
- Branch naming: `etap-N-<short-name>` (e.g. `etap-1-ingest-pipeline`).

## Development workflow per etap

Each etap 1-7 follows this sequence (see roadmap for full detail):

1. **Brainstorm** (`superpowers:brainstorming`) — only if the etap spec leaves real design ambiguity; decision saved to `DECISIONS.md`.
2. **Plan** (`superpowers:writing-plans`) — produces `docs/superpowers/plans/YYYY-MM-DD-etap-N-<name>.md`.
3. **Isolate** (`superpowers:using-git-worktrees`) — branch `etap-N-<name>`.
4. **Implement** (`superpowers:subagent-driven-development`) — fresh subagent per task, TDD (red → green → refactor), commit per task.
5. **Review** (`code-review` skill, level `medium`; `high` for Etap 5 and 6).
6. **Verify** (`superpowers:verification-before-completion`) — full test suite + manual smoke test matching DoD.
7. **Merge** (`superpowers:finishing-a-development-branch`) — squash branch into 1 commit, merge to `main`, delete branch.
8. **Document** — add entry to `DECISIONS.md` and create `docs/claude-md-fragments/etap-N.md`.
9. **Continue** — immediately start next etap without waiting for user prompt.

Gate before next etap: tests green, no blocking review findings, `DECISIONS.md` updated, smoke test confirmed.

## Architecture

```
backend/app/
  main.py          # FastAPI app + router mounts
  ingest/          # upload endpoint, Whisper transcription, chunker (Etap 1)
  search/          # semantic + BM25 + RRF fusion + reranker (Etap 2)
  agent/           # LangGraph graph definition + tools (Etap 3)
  voice/           # STT/TTS endpoints, streaming (Etap 4)

eval/
  evaluate.py      # RAG triad scorer, LLM-as-judge, report generator (Etap 5)
  golden_dataset/  # question/expected-answer pairs (source of truth)

mcp-server/        # MCP Python SDK server exposing knowledge base tools (Etap 6)

frontend/          # Next.js App Router (Etap 4)
  app/             # pages and API route handlers
  components/      # AudioRecorder, SourcePlayer (timestamp jump), ChatUI
```

**Critical data shape — chunk record (Etap 1 onward):**
```
id, episode_id, start_ts, end_ts, text, embedding (vector)
```
Chunks must align to natural speech boundaries (Whisper pauses), not character counts. `start_ts`/`end_ts` are the hard requirement for timestamp citations — do not break this invariant.

**Search modes** (Etap 2): `semantic` | `bm25` | `hybrid` | `hybrid+rerank`. RRF fuses ranks, not raw scores. Reranker is Cohere Rerank or a local cross-encoder (see `DECISIONS.md` for chosen option).

**LangGraph agent tools** (Etap 3): `search_transcripts`, `get_context_around_timestamp`, `compare_across_episodes`, `summarise_segment`. Agent uses conditional edges — step count varies per query. The DoD test explicitly verifies different step counts for simple vs. multi-hop questions.

**MCP tools** (Etap 6): `search_knowledge_base(query)`, `get_episode_summary(id)`, `find_mentions(topic)`. Return chunk + timestamp + episode — never full transcripts.

## Environment variables

Copy `.env.example` → `.env` (never commit `.env`).

| Variable | Used from |
|---|---|
| `DATABASE_URL` | Etap 1 (local: `postgresql://voicerag:voicerag@localhost:5433/voicerag`) |
| `OPENAI_API_KEY` | Etap 1 (Whisper API) |
| `ANTHROPIC_API_KEY` | Etap 3 (LangGraph LLM), Etap 5 (LLM-as-judge) |
| `ELEVENLABS_API_KEY` | Etap 4 (TTS) |
| `COHERE_API_KEY` | Etap 2 (reranking, if Cohere chosen over local cross-encoder) |

## Documentation conventions

- **`DECISIONS.md`** — every non-trivial technical decision: what was chosen, what was rejected, why. Written at end of each etap before it's considered closed. This is interview material — write as if explaining to someone who wasn't there.
- **`docs/claude-md-fragments/etap-N.md`** — per-etap additions destined for the final `CLAUDE.md`. Etap 7 merges all fragments into this file.
- Plans live in `docs/superpowers/plans/YYYY-MM-DD-etap-N-<name>.md`.

## Priority if time runs short

Do not cut: ingest + hybrid search + LangGraph agent + eval harness + MCP server.
Can simplify: voice layer (text input + TTS fallback without mic STT).
Can skip: `compare_across_episodes`, polished frontend, production deploy (Docker Compose + video demo is sufficient).
