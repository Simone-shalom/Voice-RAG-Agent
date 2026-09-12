# Etap 17 — GitHub Actions CI Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every push/PR to `main` automatically runs the full test suite (300 backend + 48 frontend tests) on GitHub, so a regression is caught before it reaches `main` instead of relying on someone remembering to run tests locally.

**Architecture:** One workflow file, `.github/workflows/ci.yml`, with two independent jobs (`backend`, `frontend`) that run in parallel on `ubuntu-latest`. Both jobs only need the repo checked out and their respective runtime installed — no database, no external API keys — because this project's entire test suite is mocked by convention (see `CLAUDE.md` "Known gotchas": *"Test suite: 244 tests, all mocked. Never require a running database or real API keys for tests."*, now 300). A status badge on `README.md` gives the human-visible signal that CI is wired up and green.

**Tech Stack:** GitHub Actions (`actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4`), Python 3.12, pytest, Node 20, vitest.

**Spec:** No separate spec doc — this was scoped and approved directly in chat during brainstorming (bounded-path task: a well-scoped addition to an already-tested, already-deployed repo, not a new subsystem). The approved design is reproduced in this plan's Goal/Architecture sections above.

## Global Constraints

- No `DATABASE_URL` or any `*_API_KEY` secrets in the workflow — the suite must pass with zero configured secrets, matching local `pytest`/`npm test` behavior today.
- Python version: 3.12 (matches `CLAUDE.md` Stack section and local dev).
- Node version: 20 (matches `CLAUDE.md` frontend test stack note: *"this project targets Node 20"*).
- Commit format: `<type>(<scope>): <subject>`, scope `etap-17`. No `Co-Authored-By` lines.
- Repo: `Simone-shalom/Voice-RAG-Agent` on GitHub (confirmed via `git remote -v`).
- Backend deps file: `backend/requirements.txt`. Frontend test script: `npm test` (→ `vitest run`, confirmed in `frontend/package.json`).

---

### Task 1: CI workflow + status badge

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (add badge line under the title, before the existing project-walkthrough/live-demo lines)

**Interfaces:**
- Consumes: nothing (first and only task — no prior task output to build on).
- Produces: a GitHub Actions workflow named `CI` with two jobs, `backend` and `frontend`, both required to pass. The badge in `README.md` references this workflow's status by its file name (`ci.yml`), so later tasks (none planned in this etap) or future etaps adding jobs to this file keep the same badge working as long as the file name and workflow `name:` don't change.

There is only one task in this plan — the "task" granularity here is a single file plus a doc line, and the two verification steps (does it fail correctly first, does it pass after) are structural, not a red/green TDD cycle in the pytest sense, since there's no application code being written. The steps below still follow strict "prove it actually runs, then prove it's correct" ordering rather than writing the finished file and hoping.

- [ ] **Step 1: Create the workflow file with the two jobs**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/ -v

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - name: Install dependencies
        run: npm ci
      - name: Run tests
        run: npm test
```

- [ ] **Step 2: Validate the YAML locally before pushing**

Run: `python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` (PyYAML is already a transitive dependency of this project's stack; if the import fails because PyYAML isn't installed in the current shell, run `pip install pyyaml` first, or equivalently open the file and confirm indentation by eye — the point of this step is to catch a YAML syntax error before spending an Actions run on it).

Expected: no exception. If it raises `yaml.scanner.ScannerError` or similar, fix the indentation and re-run.

- [ ] **Step 3: Add the status badge to README.md**

In `README.md`, immediately after the `# Voice Knowledge Agent` title line and before the one-sentence project description, insert:

```markdown
[![CI](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml)
```

So the top of the file reads:

```markdown
# Voice Knowledge Agent

[![CI](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Simone-shalom/Voice-RAG-Agent/actions/workflows/ci.yml)

Ask questions about your podcast/lecture library — by voice or text — and get spoken answers with timestamp citations you can click to jump to the exact moment in the source recording.
```

(Badge goes before the description, not before the title, to match common convention and avoid breaking the `#` header being the very first line, which some renderers rely on for the page title.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "feat(etap-17): add GitHub Actions CI pipeline for backend + frontend tests"
```

- [ ] **Step 5: Push and verify the run goes green for real**

```bash
git push
```

Then check the run in the GitHub UI or via `gh`:

```bash
gh run list --branch main --limit 1
gh run watch <run-id>   # substitute the run id from the previous command
```

Expected: both `backend` and `frontend` jobs show `success`. If either fails:
- Read the failure log (`gh run view <run-id> --log-failed`).
- Common first-run mismatches to check: a dependency in `backend/requirements.txt` that needs a system library not present on `ubuntu-latest` (unlikely — this suite runs fine in Docker already per `CLAUDE.md`'s `docker compose` setup, which is also Linux); an `npm ci` failure because `frontend/package-lock.json` is out of sync with `package.json` (fix by running `npm install` locally and committing the updated lockfile, then re-push).
- Do not silently add a secret or loosen a test to make it pass — if a test fails in CI but passes locally, that's a real environment difference worth understanding (matches this project's existing precedent of treating "passes locally, fails for real" gaps as bugs to root-cause, not to route around — see `CLAUDE.md`'s "Known gotchas" section for several examples of exactly this pattern).

- [ ] **Step 6: Record the decision in DECISIONS.md**

Append to `DECISIONS.md` (matching the existing Polish-language, `## [Etap N] Title (date)` / `**Decyzja:**` / `**Alternatywy odrzucone:**` / `**Uzasadnienie:**` format used by every prior entry — see any `## [Etap 15]` or `## [Etap 16]` entry for the exact shape). Content to cover:
- **Decyzja:** one workflow (`ci.yml`) z dwoma rownoleglymi jobami (`backend`, `frontend`), bez sekretow — cala suita jest mockowana.
- **Alternatywy odrzucone:** osobne workflow files per job (niepotrzebna fragmentacja dla dwoch jobow); uruchamianie testow w Dockerze wewnatrz CI zamiast bezposrednio na `ubuntu-latest` (wolniejsze, niepotrzebne skoro testy i tak nie potrzebuja realnej bazy).
- **Uzasadnienie:** dlaczego teraz (projekt ma 348 testow i zero automatyzacji ich uruchamiania — realna luka dla portfolio), i wszelkie faktyczne niespodzianki napotkane w Step 5 (np. lockfile mismatch, jesli wystapil) — jesli Step 5 przeszedl bez zadnych niespodzianek za pierwszym razem, napisz to wprost ("pierwszy przebieg przeszedl bez poprawek"), nie pomijaj tej informacji.

Then commit:

```bash
git add DECISIONS.md
git commit -m "docs(etap-17): record CI pipeline decision"
git push
```

## Self-Review Notes

- **Spec coverage:** workflow file (Step 1), YAML validity (Step 2), README badge (Step 3), commit (Step 4), real green-run verification (Step 5), DECISIONS.md entry (Step 6) — all approved-design elements from the brainstorming chat are covered.
- **No placeholders:** every step has literal file content or literal commands; Step 5's failure-handling guidance is deliberately a diagnostic pointer (not a placeholder) because the actual failure, if any, is unknown until the real run happens — this is inherent to "push and see," not a skipped detail.
- **Scope:** single task, matching how small this change actually is — no artificial task-splitting to mimic bigger etaps' plans.
