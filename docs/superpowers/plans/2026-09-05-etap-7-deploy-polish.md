# Deploy, README & Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project ready to show a recruiter — one-command local start, comprehensive README with architecture diagram and eval results, final CLAUDE.md compiled from all etap fragments, and a user instruction guide.

**Architecture:** No new application code. All work is documentation, docker-compose additions, and README polish. The compiled `CLAUDE.md` replaces the interim file by merging all `docs/claude-md-fragments/etap-N.md` files.

**Tech Stack:** Docker Compose, Markdown, Next.js 15 (frontend already built)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `docker-compose.yml` | Add frontend service (node:20-alpine on port 3000) |
| `README.md` | Complete portfolio README: one-liner, architecture ASCII, eval table, MCP setup, local setup, decisions, known limitations, project map |
| `CLAUDE.md` | Compiled from all `docs/claude-md-fragments/etap-*.md` — full operational guide for new Claude sessions |
| `docs/user-guide.md` | Step-by-step user instruction guide (how to ingest, query, use voice, read citations) |
| `docs/technical-spec.md` | Technical specification: system design, data flow, API reference, performance characteristics |

---

### Task 1: Docker Compose — add frontend

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add frontend service**

```yaml
frontend:
  image: node:20-alpine
  working_dir: /app
  command: sh -c "npm install && npm run dev"
  environment:
    NEXT_PUBLIC_API_URL: http://localhost:8000
  ports:
    - "3000:3000"
  depends_on:
    - backend
  volumes:
    - ./frontend:/app
    - /app/node_modules
```

- [ ] **Step 2: Verify stack starts**

```bash
docker compose up -d
curl localhost:8000/health   # {"status": "ok"}
curl localhost:3000          # HTML response
docker compose down
```

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore(deploy): add frontend service to docker-compose"
```

---

### Task 2: Final README.md

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write README with this structure**

```markdown
# Voice Knowledge Agent

[One-liner]

---

## Architecture
[ASCII diagram: Browser → FastAPI → Postgres/pgvector + LLM + TTS → Browser]

---

## Evaluation
[Table with modes × metrics — template to fill after ingesting real content]
[Instructions to run: python -m eval.evaluate ...]

---

## MCP Server
[claude_desktop_config.json snippet]
[3 tools listed]

---

## Local Setup
[docker compose up sequence]
[ingest curl example]
[agent chat curl example]

---

## Technical Decisions
[7 bullet points, linked to DECISIONS.md]

---

## Known Limitations
[4-5 bullets: no streaming, no VAD, synthetic eval dataset, no production deploy]

---

## Project Structure
[Tree of backend/ eval/ mcp-server/ frontend/]
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: write complete portfolio README"
```

---

### Task 3: Compile final CLAUDE.md

**Files:**
- Rewrite: `CLAUDE.md`

- [ ] **Step 1: Read all fragments**

```bash
cat docs/claude-md-fragments/etap-{0,1,2,3,4,5,6}.md
```

- [ ] **Step 2: Compile into sections**

Final CLAUDE.md must contain at minimum:
- Project overview + stack
- `docker compose up` + test commands
- Git conventions (no Co-Authored-By, squash policy)
- Architecture map (all directories)
- Chunk record invariant
- Search modes table
- LangGraph agent tools table
- MCP tools table
- Eval CLI usage
- Key mocking patterns for tests
- Environment variables table
- Known gotchas (ElevenLabs path, MCP venv, RRF ranks, tsvector guard)

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: compile final CLAUDE.md from all etap fragments"
```

---

### Task 4: User guide and technical spec

**Files:**
- Create: `docs/user-guide.md`
- Create: `docs/technical-spec.md`

- [ ] **Step 1: Write `docs/user-guide.md`** — step-by-step with screenshots/description:

```markdown
# How to use Voice Knowledge Agent

## 1. Start the stack
## 2. Ingest an audio file (curl or UI)
## 3. Ask a question by text
## 4. Ask a question by voice
## 5. Understanding citations (clicking timestamps)
## 6. Using the MCP tools from Claude Code
## 7. Running the eval harness
```

- [ ] **Step 2: Write `docs/technical-spec.md`** — for technical interview preparation:

```markdown
# Technical Specification

## System overview
## Data pipeline (ingest → chunk → embed → store)
## Search subsystem (semantic, BM25, RRF, reranking)
## Agent subsystem (LangGraph state machine, tools)
## Voice layer (STT → agent → TTS)
## Evaluation framework (RAG triad, LLM-as-judge)
## MCP integration
## API reference
## Performance characteristics and known limits
## Extension points
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide.md docs/technical-spec.md
git commit -m "docs: add user guide and technical specification"
```

---

### Task 5: Squash and final verification

- [ ] **Step 1: Verify full stack starts cleanly**

```bash
docker compose up -d
sleep 10
curl localhost:8000/health  # {"status": "ok"}
curl localhost:3000 -I      # 200 OK
docker compose down
```

- [ ] **Step 2: Run tests one final time**

```bash
cd backend && python -m pytest tests/ -v
# Expected: 131 passed
```

- [ ] **Step 3: Squash-commit to main**

```bash
git reset --soft $(git merge-base HEAD main)
git commit -m "feat(etap-7): final README, compiled CLAUDE.md, user guide, technical spec"
```

- [ ] **Step 4: Verify git log**

```bash
git log --oneline
# Should show 7 feat(etap-N) commits + chore bootstrap + docs commits
```
