# Etap 16 — Diaryzacja mówców + auto-summary/chapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** citations say "Speaker 0 said..." (or a real name, best-effort) instead of an anonymous fragment, and every episode has a summary and a clickable chapter list before anyone asks a question.

**Architecture:** Two new ingest-pipeline steps run synchronously after chunking/embedding, before the ingest response returns: `diarizer.py` calls Deepgram's prerecorded API (`diarize=true`) on the same audio file already used for transcription, and maps Deepgram's word-level speaker tags onto our existing chunks by time-overlap; `summarizer.py` makes one Claude Haiku tool-use call over the full chunked transcript to produce a summary, chapter markers, and a best-effort `Speaker N → real name` mapping in a single structured response. Both failure modes degrade gracefully — a missing API key or a timeout leaves `speaker_label`/`summary`/`chapters` `NULL`, never aborts the ingest. `speaker_label` flows from `Chunk` through the existing search functions into the agent's citations and the frontend's `SourcePlayer`; `summary`/`chapters` flow from `Episode` into the episode-list API and a new expanded panel in the sidebar.

**Tech Stack:** Deepgram prerecorded REST API (`DEEPGRAM_API_KEY`, reused from Etap 15 — same account, no new provider), raw `httpx` (matching this project's established "no SDK for third-party HTTP APIs" convention from `voice/tts.py`), `anthropic` Python SDK direct (matching `eval/scorer.py`'s pattern, not the LangChain wrapper) with tool-use/function-calling to force structured JSON output, PostgreSQL (two new nullable columns on `chunks`, two new nullable columns on `episodes`, same idempotent-migration pattern as Etap 13's `text_search` column).

**Spec:** `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md` (section "Etap 16 — Diaryzacja mówców + auto-summary/chapters per odcinek")

## Global Constraints

- No `Co-Authored-By: Claude Code` in commits; commit format `<type>(<scope>): <subject>`, scope `ingest`, `search`, `agent`, or `frontend`.
- Backend tests are 100% mocked — never require a running database, a real Deepgram/Anthropic key, or a network call. `DATABASE_URL`/`OPENAI_API_KEY` are set via `os.environ.setdefault` at the top of every test file per this project's existing convention (see any `backend/tests/test_ingest_*.py` file) — do not require `DEEPGRAM_API_KEY`/`ANTHROPIC_API_KEY` to be set for tests to pass; missing-key behavior (a caught `RuntimeError`) IS the thing several tests verify.
- Diarization and summarization failures (missing key, timeout, malformed response) must never abort ingest — the DoD requires an episode ingested with a diarization/summarization outage to still be fully searchable, just with `speaker_label`/`summary`/`chapters` left `NULL`.
- Speaker→chunk mapping must use an explicit >50% overlap-of-chunk-duration threshold — never guess on an ambiguous/overlapping-speech boundary; below threshold, `speaker_label = None`.
- Chapter/summary structured output must come from the LLM's tool-use/function-calling mechanism, never from regex-parsing free text — this project has been bitten twice already by trusting an assumed wire/output shape instead of verifying it (see `CLAUDE.md`'s "Known gotchas": the Anthropic streaming content-block bug and the SQLAlchemy `:param::type` bug).
- No voice fingerprinting across episodes, no manual speaker-label editing UI — explicitly out of scope per the spec's Nie-cele.

---

### Task 1: DB schema — `Chunk.speaker_label`, `Episode.summary`, `Episode.chapters`

**Files:**
- Modify: `backend/app/db/models.py`
- Modify: `backend/app/db/session.py`
- Test: `backend/tests/test_session.py`

**Interfaces:**
- Produces: `Chunk.speaker_label: str | None`, `Episode.summary: str | None`, `Episode.chapters: list[dict] | None` (JSON column, same pattern as the existing `ChatMessage.sources` JSON column) — every later task that creates/reads these ORM objects relies on these exact attribute names and nullability.

This task establishes the schema before any ingest-pipeline code touches it. Follows the exact idempotent-`ALTER TABLE`-in-`init_db()` pattern this project already uses for `chunks.text_search` (Etap 13) — `Base.metadata.create_all()` only creates tables that don't exist yet; it never alters an existing production table, so a manual, idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS`-equivalent is required for a column added to a model that's already deployed.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_session.py`:

```python
def test_init_db_adds_speaker_label_and_summary_columns():
    """
    Regression test for the same class of bug test_init_db_creates_tables_before_altering_chunks
    guards: Base.metadata.create_all() only creates missing tables, it never alters
    an existing chunks/episodes table already deployed without these columns — an
    idempotent ALTER TABLE is required, same as the text_search column (Etap 13).
    """
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    executed_sql = []
    mock_conn.execute.side_effect = lambda stmt, *a, **k: executed_sql.append(str(stmt))

    with patch("app.db.models.Base"):
        init_db(mock_engine)

    combined = "\n".join(executed_sql)
    assert "speaker_label" in combined
    assert "ADD COLUMN" in combined and "speaker_label" in combined
    assert "summary" in combined
    assert "chapters" in combined
```

Add `from unittest.mock import patch` to the top of the test file if not already imported (it already is, per the existing `test_init_db_sql_contains_tsvector` test in this file).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_session.py::test_init_db_adds_speaker_label_and_summary_columns -v`
Expected: FAIL — `speaker_label`/`summary`/`chapters` not found in any executed SQL, since `init_db` doesn't emit those statements yet.

- [ ] **Step 3: Add the ORM columns**

In `backend/app/db/models.py`, add `speaker_label` to `Chunk`:

```python
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(UUID(as_uuid=True), ForeignKey("episodes.id"), nullable=False)
    start_ts = Column(Float, nullable=False)
    end_ts = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    speaker_label = Column(String, nullable=True)
    episode = relationship("Episode", back_populates="chunks")
```

Add `summary`/`chapters` to `Episode`:

```python
class Episode(Base):
    __tablename__ = "episodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    chapters = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    chunks = relationship("Chunk", back_populates="episode", lazy="select")
```

(`JSON` is already imported in this file — used by `ChatMessage.sources`.)

- [ ] **Step 4: Add the idempotent migration to `init_db`**

In `backend/app/db/session.py`, inside `init_db`, after the existing `text_search` `DO $$ ... $$` block and its `CREATE INDEX IF NOT EXISTS` (still inside the same `with engine_.connect() as conn:` block, before the final `conn.commit()`), add:

```python
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'speaker_label'
                  ) THEN
                    ALTER TABLE chunks ADD COLUMN speaker_label VARCHAR;
                  END IF;
                  IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'episodes' AND column_name = 'summary'
                  ) THEN
                    ALTER TABLE episodes ADD COLUMN summary TEXT;
                  END IF;
                  IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'episodes' AND column_name = 'chapters'
                  ) THEN
                    ALTER TABLE episodes ADD COLUMN chapters JSON;
                  END IF;
                END
                $$
                """
            )
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_session.py -v`
Expected: all tests in this file pass, including the new one and the pre-existing `test_init_db_creates_tables_before_altering_chunks` (unaffected — this new block runs after `create_all`, same as the `text_search` block it sits beside).

- [ ] **Step 6: Run the full suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all 268 tests pass, 0 regressions (no other code references these new columns yet).

- [ ] **Step 7: Commit**

```bash
git add backend/app/db/models.py backend/app/db/session.py backend/tests/test_session.py
git commit -m "feat(ingest): add speaker_label, summary, chapters columns (Etap 16 schema)

Chunk.speaker_label (nullable) and Episode.summary/chapters (nullable)
follow the same idempotent ALTER TABLE pattern as Etap 13's text_search
column — Base.metadata.create_all() never alters an already-deployed
table, only creates missing ones."
```

---

### Task 2: `diarizer.py` — Deepgram prerecorded diarization + speaker→chunk mapping

**Files:**
- Create: `backend/app/ingest/diarizer.py`
- Test: `backend/tests/test_diarizer.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (pure functions operate on plain dicts, not ORM objects) — but its output (`speaker_label: str | None` per chunk) is what Task 4 assigns onto the `Chunk.speaker_label` column Task 1 added.
- Produces:
  - `diarize_audio(audio_path: Path) -> list[dict]` — calls Deepgram's prerecorded API with `diarize=true`, returns word-level entries `[{"start": float, "end": float, "speaker": int}, ...]`. Raises `RuntimeError` (message contains `"DEEPGRAM_API_KEY"`) if the key is unset.
  - `merge_into_speaker_segments(words: list[dict]) -> list[dict]` — collapses consecutive same-speaker words into `[{"start_ts": float, "end_ts": float, "speaker": int}, ...]`.
  - `assign_speaker_labels(chunks: list[dict], words: list[dict], threshold: float = 0.5) -> list[str | None]` — for each chunk (dict with `start_ts`/`end_ts`), returns `"Speaker N"` if that speaker covers more than `threshold` of the chunk's duration, else `None`. Returns a list parallel to (same length, same order as) `chunks`.

**IMPORTANT — verify before trusting this:** the response shape below (`results.channels[0].alternatives[0].words[]`, each word carrying a `speaker` integer) is Deepgram's documented prerecorded-API diarization shape as of this plan's writing. This project has shipped two bugs already purely because a wire shape from documentation didn't match what mocked tests assumed (Anthropic's streaming `content` being a list not a string; SQLAlchemy's `:param::type` bind-param bug) — and Etap 15 flagged the exact same risk for Deepgram's *streaming* API and left it for live verification. Before trusting this in production, run a real diarized transcript through `diarize_audio` with a genuine `DEEPGRAM_API_KEY` against a short multi-speaker clip and print the raw JSON — if the shape differs, fix `diarize_audio`'s parsing and add a regression test using the real captured shape, the same way `test_ignores_tool_use_blocks_interleaved_with_text` was added after the Anthropic bug. This live check belongs in Task 10 alongside its own live-verification step, not blocking this task's merge.

- [ ] **Step 1: Write the failing tests**

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest.diarizer import assign_speaker_labels, diarize_audio, merge_into_speaker_segments


def test_diarize_audio_raises_without_api_key(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            diarize_audio(audio_file)


def test_diarize_audio_parses_word_level_speakers(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": 0},
                                {"word": "there", "start": 0.5, "end": 1.0, "speaker": 0},
                                {"word": "hi", "start": 1.2, "end": 1.5, "speaker": 1},
                            ]
                        }
                    ]
                }
            ]
        }
    }

    with patch("app.ingest.diarizer.httpx.post", return_value=mock_response) as mock_post, \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}):
        words = diarize_audio(audio_file)

    assert words == [
        {"start": 0.0, "end": 0.5, "speaker": 0},
        {"start": 0.5, "end": 1.0, "speaker": 0},
        {"start": 1.2, "end": 1.5, "speaker": 1},
    ]
    call_kwargs = mock_post.call_args
    assert "diarize=true" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["Authorization"] == "Token test-key"


def test_merge_into_speaker_segments_collapses_consecutive_same_speaker():
    words = [
        {"start": 0.0, "end": 0.5, "speaker": 0},
        {"start": 0.5, "end": 1.0, "speaker": 0},
        {"start": 1.2, "end": 1.5, "speaker": 1},
        {"start": 1.5, "end": 2.0, "speaker": 1},
        {"start": 2.1, "end": 2.5, "speaker": 0},
    ]
    segments = merge_into_speaker_segments(words)
    assert segments == [
        {"start_ts": 0.0, "end_ts": 1.0, "speaker": 0},
        {"start_ts": 1.2, "end_ts": 2.0, "speaker": 1},
        {"start_ts": 2.1, "end_ts": 2.5, "speaker": 0},
    ]


def test_merge_into_speaker_segments_empty_input():
    assert merge_into_speaker_segments([]) == []


def test_assign_speaker_labels_picks_majority_overlap_speaker():
    chunks = [{"start_ts": 0.0, "end_ts": 2.0, "text": "hello there"}]
    # Speaker 0 covers [0.0, 1.5] = 1.5s of this 2.0s chunk = 75% > 50% threshold
    words = [
        {"start": 0.0, "end": 1.5, "speaker": 0},
        {"start": 1.5, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == ["Speaker 0"]


def test_assign_speaker_labels_returns_none_below_threshold():
    chunks = [{"start_ts": 0.0, "end_ts": 2.0, "text": "hello there"}]
    # Speaker 0 covers exactly 1.0s of this 2.0s chunk = 50%, NOT strictly > 50%
    words = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == [None]


def test_assign_speaker_labels_returns_none_with_no_overlapping_words():
    chunks = [{"start_ts": 10.0, "end_ts": 12.0, "text": "hello"}]
    words = [{"start": 0.0, "end": 1.0, "speaker": 0}]
    labels = assign_speaker_labels(chunks, words)
    assert labels == [None]


def test_assign_speaker_labels_handles_multiple_chunks_in_order():
    chunks = [
        {"start_ts": 0.0, "end_ts": 1.0, "text": "a"},
        {"start_ts": 1.0, "end_ts": 2.0, "text": "b"},
    ]
    words = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
    ]
    labels = assign_speaker_labels(chunks, words)
    assert labels == ["Speaker 0", "Speaker 1"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_diarizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest.diarizer'`.

- [ ] **Step 3: Implement**

```python
import os
from pathlib import Path

import httpx

DEEPGRAM_PRERECORDED_URL = (
    "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true&punctuate=true"
)


def diarize_audio(audio_path: Path) -> list[dict]:
    """
    Calls Deepgram's prerecorded API with diarize=true on the given audio
    file (the same file already transcribed by Whisper for chunking).
    Returns word-level entries: [{"start": float, "end": float, "speaker": int}, ...].
    Raises RuntimeError if DEEPGRAM_API_KEY is unset.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set — cannot diarize episode")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = httpx.post(
        DEEPGRAM_PRERECORDED_URL,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/*"},
        content=audio_bytes,
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    words = data["results"]["channels"][0]["alternatives"][0]["words"]
    return [{"start": w["start"], "end": w["end"], "speaker": w["speaker"]} for w in words]


def merge_into_speaker_segments(words: list[dict]) -> list[dict]:
    """Collapse consecutive same-speaker words into segments: [{start_ts, end_ts, speaker}, ...]."""
    if not words:
        return []
    segments: list[dict] = []
    current = {"start_ts": words[0]["start"], "end_ts": words[0]["end"], "speaker": words[0]["speaker"]}
    for w in words[1:]:
        if w["speaker"] == current["speaker"]:
            current["end_ts"] = w["end"]
        else:
            segments.append(current)
            current = {"start_ts": w["start"], "end_ts": w["end"], "speaker": w["speaker"]}
    segments.append(current)
    return segments


def assign_speaker_labels(
    chunks: list[dict], words: list[dict], threshold: float = 0.5
) -> list[str | None]:
    """
    For each chunk (dict with start_ts/end_ts), returns the speaker label
    ("Speaker N") whose total overlap covers more than `threshold` of the
    chunk's duration, else None — an ambiguous or no-overlap chunk is never
    guessed at, per this etap's explicit design (overlapping/interrupting
    speech near a chunk boundary can otherwise misassign a speaker).
    """
    segments = merge_into_speaker_segments(words)
    labels: list[str | None] = []
    for chunk in chunks:
        chunk_start, chunk_end = chunk["start_ts"], chunk["end_ts"]
        duration = chunk_end - chunk_start
        if duration <= 0:
            labels.append(None)
            continue
        overlap_by_speaker: dict[int, float] = {}
        for seg in segments:
            overlap = min(chunk_end, seg["end_ts"]) - max(chunk_start, seg["start_ts"])
            if overlap > 0:
                overlap_by_speaker[seg["speaker"]] = overlap_by_speaker.get(seg["speaker"], 0.0) + overlap
        if not overlap_by_speaker:
            labels.append(None)
            continue
        best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])
        labels.append(f"Speaker {best_speaker}" if best_overlap / duration > threshold else None)
    return labels
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_diarizer.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/diarizer.py backend/tests/test_diarizer.py
git commit -m "feat(ingest): Deepgram prerecorded diarization + speaker-to-chunk mapping

Word-level speaker tags from Deepgram's prerecorded API (diarize=true,
same account as Etap 15's streaming STT) get collapsed into speaker
segments and mapped onto our existing chunks by time-overlap — a chunk
gets a speaker_label only when one speaker covers >50% of its duration,
otherwise NULL rather than guessing on an ambiguous boundary. Message
shape needs live verification (see docstring) before Task 10 ships —
this project has shipped broken streaming/wire-shape assumptions from
documentation twice already."
```

---

### Task 3: `summarizer.py` — structured summary + chapters + speaker-name mapping

**Files:**
- Create: `backend/app/ingest/summarizer.py`
- Test: `backend/tests/test_summarizer.py`
- Modify: `backend/requirements.txt` (add `anthropic==1.4.0`)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (operates on the same `chunk_dicts` shape `chunk_segments()` already produces: `{"start_ts": float, "end_ts": float, "text": str}`).
- Produces: `summarize_episode(chunks: list[dict]) -> dict` — one Claude Haiku tool-use call, returns `{"summary": str, "chapters": [{"start_ts": float, "title": str}, ...], "speaker_names": {"Speaker N": str, ...}}`. Raises `RuntimeError` (message contains `"ANTHROPIC_API_KEY"`) if the key is unset. `chapters` and `speaker_names` are always present (possibly empty list/dict), never missing keys — Task 4 relies on this to avoid `.get()` defensive code at every call site.

- [ ] **Step 1: Add the dependency**

In `backend/requirements.txt`, add a new line: `anthropic==1.4.0`

(This version is already the one resolved transitively by `langchain-anthropic==1.7.1` in this project's current environment — pinning it explicitly, rather than relying on the transitive resolution, avoids the exact class of surprise Etap 15 hit with an under-specified `websockets` pin.)

- [ ] **Step 2: Write the failing tests**

```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from unittest.mock import MagicMock, patch

import pytest

from app.ingest.summarizer import summarize_episode


def _make_tool_use_response(input_payload: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_payload
    message = MagicMock()
    message.content = [block]
    return message


def test_summarize_episode_raises_without_api_key():
    chunks = [{"start_ts": 0.0, "end_ts": 5.0, "text": "hello"}]
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            summarize_episode(chunks)


def test_summarize_episode_returns_structured_output():
    chunks = [
        {"start_ts": 0.0, "end_ts": 10.0, "text": "Welcome to the show, I'm Gary Jordan."},
        {"start_ts": 10.0, "end_ts": 60.0, "text": "Today we discuss NASA's new rocket."},
    ]
    payload = {
        "summary": "Gary Jordan hosts a discussion about NASA's new rocket program.",
        "chapters": [{"start_ts": 0.0, "title": "Introduction"}, {"start_ts": 10.0, "title": "NASA's new rocket"}],
        "speaker_names": {"Speaker 0": "Gary Jordan"},
    }
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = summarize_episode(chunks)

    assert result == payload
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "emit_episode_summary"}
    assert call_kwargs["tools"][0]["name"] == "emit_episode_summary"


def test_summarize_episode_empty_chunks_returns_empty_structure_without_api_call():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = summarize_episode([])
    assert result == {"summary": "", "chapters": [], "speaker_names": {}}


def test_summarize_episode_truncates_long_transcripts():
    long_text = "word " * 5000  # far exceeds MAX_TRANSCRIPT_CHARS
    chunks = [{"start_ts": 0.0, "end_ts": 10.0, "text": long_text}]
    payload = {"summary": "s", "chapters": [], "speaker_names": {}}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_tool_use_response(payload)

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        summarize_episode(chunks)

    sent_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[...transcript truncated...]" in sent_content


def test_summarize_episode_raises_if_no_tool_use_block_returned():
    chunks = [{"start_ts": 0.0, "end_ts": 5.0, "text": "hello"}]
    text_block = MagicMock()
    text_block.type = "text"
    message = MagicMock()
    message.content = [text_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = message

    with patch("app.ingest.summarizer._client", return_value=mock_client), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with pytest.raises(RuntimeError, match="structured summary"):
            summarize_episode(chunks)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && pip install -r requirements.txt && python -m pytest tests/test_summarizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest.summarizer'`.

- [ ] **Step 4: Implement**

```python
import os
from functools import lru_cache

import anthropic

SUMMARIZER_MODEL = "claude-haiku-4-5-20251001"
MAX_TRANSCRIPT_CHARS = 12000  # ~ a 30-45 min podcast's worth of chunked transcript

_SUMMARY_TOOL = {
    "name": "emit_episode_summary",
    "description": "Emit a structured summary of a podcast episode transcript.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A 2-4 sentence summary of the episode's content.",
            },
            "chapters": {
                "type": "array",
                "description": "Chapter markers at natural topic-change points in the episode.",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_ts": {"type": "number", "description": "Start time in seconds"},
                        "title": {"type": "string", "description": "Short chapter title"},
                    },
                    "required": ["start_ts", "title"],
                },
            },
            "speaker_names": {
                "type": "object",
                "description": (
                    "Best-effort mapping from 'Speaker N' labels to real names actually "
                    "stated in the transcript (e.g. someone saying 'I'm Gary Jordan, your "
                    "host'). Omit any speaker whose real name is never stated — never guess."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["summary", "chapters", "speaker_names"],
    },
}


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot summarize episode")
    return anthropic.Anthropic(api_key=api_key)


def _format_transcript(chunks: list[dict]) -> str:
    lines = [f"[{c['start_ts']:.1f}s] {c['text']}" for c in chunks]
    text = "\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n[...transcript truncated...]"
    return text


def summarize_episode(chunks: list[dict]) -> dict:
    """
    One Claude Haiku call over the chunked transcript, forced into structured
    output via tool-use (never parses free text with regex — free-text
    structured output is known to be unstable in this exact way, see
    CLAUDE.md's gotchas for two prior incidents of trusting an assumed shape).

    Returns {"summary": str, "chapters": [{"start_ts": float, "title": str}, ...],
    "speaker_names": {"Speaker N": str, ...}} — chapters/speaker_names are
    always present, possibly empty. Raises RuntimeError if ANTHROPIC_API_KEY
    is unset, or if the model doesn't return the expected tool-use block.
    """
    if not chunks:
        return {"summary": "", "chapters": [], "speaker_names": {}}

    transcript = _format_transcript(chunks)
    message = _client().messages.create(
        model=SUMMARIZER_MODEL,
        max_tokens=1024,
        tools=[_SUMMARY_TOOL],
        tool_choice={"type": "tool", "name": "emit_episode_summary"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this podcast episode transcript. Each line is "
                    "timestamped in seconds from the start of the episode.\n\n"
                    f"{transcript}"
                ),
            }
        ],
    )
    for block in message.content:
        if block.type == "tool_use":
            return {
                "summary": block.input.get("summary", ""),
                "chapters": block.input.get("chapters", []),
                "speaker_names": block.input.get("speaker_names", {}),
            }
    raise RuntimeError("Claude did not return a structured summary")
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_summarizer.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ingest/summarizer.py backend/tests/test_summarizer.py backend/requirements.txt
git commit -m "feat(ingest): episode summary + chapters + speaker-name mapping via Claude Haiku

Single tool-use call over the chunked transcript, forced into a fixed
JSON schema (summary, chapters, speaker_names) — never regex-parses
free text, per this project's own established lesson about trusting
an assumed LLM/API output shape without enforcement."
```

---

### Task 4: Integrate diarization + summarization into `ingest_audio()`

**Files:**
- Modify: `backend/app/ingest/service.py`
- Test: `backend/tests/test_ingest_service.py`

**Interfaces:**
- Consumes: `diarize_audio`, `assign_speaker_labels` (Task 2); `summarize_episode` (Task 3); `Chunk.speaker_label`, `Episode.summary`, `Episode.chapters` (Task 1).
- Produces: `ingest_audio()`'s behavior is otherwise unchanged (same signature, same return type `Episode`) — this task only adds two internal steps and two new fields on the objects it already creates.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_ingest_service.py` (the existing three tests in this file are untouched — they don't set `DEEPGRAM_API_KEY`/`ANTHROPIC_API_KEY`, so the new code's missing-key `RuntimeError` gets caught internally and those tests keep passing unchanged, which is itself a regression test for the graceful-degradation requirement):

```python
def test_ingest_audio_sets_speaker_label_and_summary_on_success(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    mock_segments = [{"text": "Hello world.", "start": 0.0, "end": 2.0}]
    mock_words = [{"start": 0.0, "end": 2.0, "speaker": 0}]
    mock_summary_result = {
        "summary": "A short greeting.",
        "chapters": [{"start_ts": 0.0, "title": "Greeting"}],
        "speaker_names": {"Speaker 0": "Alex"},
    }

    with patch("app.ingest.service.transcribe", return_value=mock_segments), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=mock_words), \
         patch("app.ingest.service.summarize_episode", return_value=mock_summary_result):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_episode = db.add.call_args_list[0][0][0]
    added_chunk = db.add.call_args_list[1][0][0]
    assert added_episode.summary == "A short greeting."
    assert added_episode.chapters == [{"start_ts": 0.0, "title": "Greeting"}]
    # Speaker 0's overlap covers the whole 2.0s chunk (100% > 50% threshold),
    # and speaker_names remaps "Speaker 0" -> "Alex" from the same summarizer call.
    assert added_chunk.speaker_label == "Alex"


def test_ingest_audio_survives_diarization_failure(tmp_path):
    """
    DoD requirement: a diarization outage (bad key/timeout) must leave the
    episode fully ingested and searchable with speaker_label=None, not crash
    the ingest.
    """
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", side_effect=RuntimeError("DEEPGRAM_API_KEY not set")), \
         patch("app.ingest.service.summarize_episode", return_value={"summary": "", "chapters": [], "speaker_names": {}}):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_chunk = db.add.call_args_list[1][0][0]
    assert added_chunk.speaker_label is None
    db.commit.assert_called_once()


def test_ingest_audio_survives_summarization_failure(tmp_path):
    """DoD requirement: a summarization outage must leave summary/chapters=None, not crash."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=[]), \
         patch("app.ingest.service.summarize_episode", side_effect=RuntimeError("ANTHROPIC_API_KEY not set")):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_episode = db.add.call_args_list[0][0][0]
    assert added_episode.summary is None
    assert added_episode.chapters is None
    db.commit.assert_called_once()


def test_ingest_audio_speaker_label_stays_generic_without_name_mapping(tmp_path):
    """When summarize_episode's speaker_names has no entry for a Speaker N label, keep the generic label."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")
    db = make_mock_db()

    with patch("app.ingest.service.transcribe", return_value=[{"text": "Hi.", "start": 0.0, "end": 1.0}]), \
         patch("app.ingest.service.embed_texts", return_value=[[0.1] * 1536]), \
         patch("app.ingest.service.diarize_audio", return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}]), \
         patch("app.ingest.service.summarize_episode", return_value={"summary": "s", "chapters": [], "speaker_names": {}}):
        ingest_audio(audio_file, "test.mp3", None, db)

    added_chunk = db.add.call_args_list[1][0][0]
    assert added_chunk.speaker_label == "Speaker 0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_ingest_service.py -v`
Expected: FAIL — `ingest_audio` doesn't call `diarize_audio`/`summarize_episode` yet, so `added_chunk.speaker_label`/`added_episode.summary` don't exist as meaningful assertions (they'll be whatever the `Chunk`/`Episode` constructor defaults to, likely absent-keyword `AttributeError` won't occur since `MagicMock` isn't involved here — these are real `Chunk`/`Episode` ORM instances — the assertions will just fail with `None != "Alex"` etc., or `speaker_label` attribute simply not set on construction).

- [ ] **Step 3: Implement**

Modify `backend/app/ingest/service.py`:

```python
import logging
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from .chunker import chunk_segments
from .diarizer import assign_speaker_labels, diarize_audio
from .embedder import embed_texts
from .rss import parse_feed
from .summarizer import summarize_episode
from .transcriber import transcribe
from ..core.url_safety import assert_safe_url
from ..db.models import Chunk, Episode

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB cap on ingest_from_url downloads


def ingest_audio(audio_path: Path, filename: str, source_url: str | None, db: Session) -> Episode:
    segments = transcribe(audio_path)
    chunk_dicts = chunk_segments(segments)
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)

    speaker_labels: list[str | None] = [None] * len(chunk_dicts)
    try:
        words = diarize_audio(audio_path)
        speaker_labels = assign_speaker_labels(chunk_dicts, words)
    except Exception:
        # Diarization is a nice-to-have, never a reason to fail an ingest —
        # the episode must stay fully searchable even if Deepgram is down.
        logger.exception("diarization failed — continuing without speaker labels")

    summary: str | None = None
    chapters: list[dict] | None = None
    try:
        summary_result = summarize_episode(chunk_dicts)
        summary = summary_result["summary"] or None
        chapters = summary_result["chapters"] or None
        speaker_names = summary_result["speaker_names"]
        if speaker_names:
            speaker_labels = [speaker_names.get(label, label) if label else None for label in speaker_labels]
    except Exception:
        # Same rationale as diarization above — summarization must never
        # abort an otherwise-successful ingest.
        logger.exception("summarization failed — continuing without summary/chapters")

    episode = Episode(id=uuid.uuid4(), filename=filename, source_url=source_url, summary=summary, chapters=chapters)
    db.add(episode)

    for chunk_dict, embedding, speaker_label in zip(chunk_dicts, embeddings, speaker_labels):
        db.add(
            Chunk(
                episode_id=episode.id,
                start_ts=chunk_dict["start_ts"],
                end_ts=chunk_dict["end_ts"],
                text=chunk_dict["text"],
                embedding=embedding,
                speaker_label=speaker_label,
            )
        )

    db.commit()
    db.refresh(episode)
    return episode
```

(Only `ingest_audio` changes — `ingest_from_url` and `ingest_from_rss` are untouched, since they both funnel through `ingest_audio`.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_ingest_service.py -v`
Expected: all 7 tests pass (3 pre-existing + 4 new).

- [ ] **Step 5: Run the full suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass, 0 regressions. `test_ingest_rss.py`/`test_ingest_url.py`/`test_ingest_router.py` all mock `ingest_audio` itself or mock `transcribe`/`embed_texts` without mocking `diarize_audio`/`summarize_episode` — check these files: if any of them call the REAL `ingest_audio` (not a mock of `ingest_audio` itself) without `DEEPGRAM_API_KEY`/`ANTHROPIC_API_KEY` set, the new code paths will raise `RuntimeError` internally and be caught (per the try/except above) — this should be transparent and not break anything, but confirm by actually running the suite rather than assuming.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ingest/service.py backend/tests/test_ingest_service.py
git commit -m "feat(ingest): wire diarization + summarization into ingest_audio

Both run synchronously after chunking/embedding, before commit. Each
is independently wrapped so a diarization or summarization failure
(bad key, timeout) never aborts the ingest — the episode stays fully
searchable with speaker_label/summary/chapters left NULL, per this
etap's explicit DoD requirement. Speaker names from the summarizer's
best-effort speaker_names mapping remap the diarizer's generic
'Speaker N' labels when a real name was actually stated."
```

---

### Task 5: Search layer — surface `speaker_label` in search results

**Files:**
- Modify: `backend/app/search/searcher.py`
- Modify: `backend/app/search/bm25.py`
- Test: `backend/tests/test_searcher.py`

**Interfaces:**
- Consumes: `Chunk.speaker_label` (Task 1) via raw SQL `SELECT`.
- Produces: `semantic_search`/`bm25_search`/`hybrid_search`/`hybrid_rerank_search` (unchanged signatures) now each chunk dict additionally carries `"speaker_label": str | None`. `reciprocal_rank_fusion` (in `rrf.py`) and `rerank` (in `reranker.py`) both spread `**chunk`/`**chunks[r.index]` already, so they need NO changes — `speaker_label` passes through automatically once it's in the two base search functions' output.

**IMPORTANT:** the existing test `test_semantic_search_returns_chunk_dicts` in `backend/tests/test_searcher.py` asserts an EXACT key set: `assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity"}`. Adding `speaker_label` will break this exact-match assertion — update it as part of this task, don't leave it failing.

- [ ] **Step 1: Update the failing test first**

In `backend/tests/test_searcher.py`, update `make_mock_row` to accept and set `speaker_label`:

```python
def make_mock_row(chunk_id="c1", episode_id="e1", start_ts=1.0, end_ts=5.0, text="Hello", similarity=0.9, speaker_label=None):
    row = MagicMock()
    row.id = chunk_id
    row.episode_id = episode_id
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.similarity = similarity
    row.speaker_label = speaker_label
    return row
```

Update the exact-key-set assertion:

```python
def test_semantic_search_returns_chunk_dicts():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        make_mock_row(),
        make_mock_row(chunk_id="c2", similarity=0.8),
    ]

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        results = semantic_search("test query", limit=10, db=db)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity"] == 0.9
    assert set(results[0].keys()) == {"chunk_id", "episode_id", "start_ts", "end_ts", "text", "similarity", "speaker_label"}
```

Add a new test:

```python
def test_semantic_search_returns_speaker_label_when_present():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [make_mock_row(speaker_label="Speaker 0")]

    with patch("app.search.searcher.embed_texts", return_value=[[0.1] * 1536]):
        results = semantic_search("test query", limit=10, db=db)

    assert results[0]["speaker_label"] == "Speaker 0"
```

Also add a `backend/tests/test_bm25.py` test if one doesn't already exist for this exact case — check for an existing `test_bm25.py` first (search this repo) and follow its existing row-mocking pattern; add an equivalent `speaker_label`-presence test there. If `test_bm25.py` doesn't exist yet, skip adding one — `bm25_search` already has no dedicated test file per this codebase's current structure, and adding one is out of this task's scope (only the two SELECT/dict changes are required).

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_searcher.py -v`
Expected: FAIL — the updated exact-key-set assertion and the new `speaker_label` assertion both fail since `semantic_search` doesn't select/return that field yet.

- [ ] **Step 3: Implement**

In `backend/app/search/searcher.py`, update `semantic_search`'s SQL and returned dict:

```python
def semantic_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Embed the query and find the nearest chunks by cosine similarity.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity, speaker_label}.
    """
    embedding = embed_texts([query])[0]
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text, c.speaker_label,
                   1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:emb AS vector)
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
            "speaker_label": r.speaker_label,
        }
        for r in rows
    ]
```

In `backend/app/search/bm25.py`, the same change:

```python
def bm25_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Full-text search using PostgreSQL tsvector/ts_rank.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity, speaker_label}
    where similarity is the ts_rank score (higher = more relevant).
    """
    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text, c.speaker_label,
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
            "speaker_label": r.speaker_label,
        }
        for r in rows
    ]
```

Update the docstrings of `hybrid_search` and `hybrid_rerank_search` in `searcher.py` (no code change needed there — only the docstrings' field lists, since `speaker_label` now flows through automatically):

```python
def hybrid_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Fuse semantic and BM25 results via Reciprocal Rank Fusion.
    Fetches 2× limit from each sub-searcher for fusion headroom.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, speaker_label, similarity}
    where similarity is the RRF score.
    """
```

```python
def hybrid_rerank_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Hybrid RRF search followed by Cohere reranking.
    Returns the same fields as hybrid_search, reordered/filtered by relevance.
    Raises RuntimeError if COHERE_API_KEY is not set.
    """
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_searcher.py -v`
Expected: all tests pass, including the updated exact-key-set assertion.

- [ ] **Step 5: Run the full suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all tests pass. Check `test_search_router.py` and any other file mocking `semantic_search`/`bm25_search`/`hybrid_search` return values — if any of those tests assert an exact key set on a chunk dict the way `test_searcher.py` did, they need the same one-line fix; if they only check specific keys' values (not exact set equality), they're unaffected. Fix any that break.

- [ ] **Step 6: Commit**

```bash
git add backend/app/search/searcher.py backend/app/search/bm25.py backend/tests/test_searcher.py
git commit -m "feat(search): surface Chunk.speaker_label in search results

semantic_search and bm25_search both select+return speaker_label now;
hybrid_search/hybrid_rerank_search need no code change since RRF fusion
and Cohere reranking both spread the full chunk dict through already."
```

---

### Task 6: Agent citations carry `speaker_label`

**Files:**
- Modify: `backend/app/agent/streaming.py`
- Test: `backend/tests/agent/test_streaming.py`

**Interfaces:**
- Consumes: `speaker_label` field in chunk dicts from `sources_sink` (Task 5's search functions populate this — `sources_sink` is filled by `agent/tools.py`, which just extends/appends whatever `search_fn` returns, unchanged).
- Produces: `stream_agent_response`'s `"sources"` event dicts now include `"speaker_label": str | None` per source, in addition to the existing `episode_id`/`start_ts`/`end_ts`/`text`.

- [ ] **Step 1: Write the failing test**

In `backend/tests/agent/test_streaming.py`, find the existing sources-related tests (search for `MAX_SOURCES` or `"sources"` in this file) and add:

```python
@pytest.mark.asyncio
async def test_sources_event_includes_speaker_label():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Answer.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append({
                "episode_id": "e1", "start_ts": 1.0, "end_ts": 5.0,
                "text": "hello", "speaker_label": "Speaker 0", "chunk_id": "c1", "similarity": 0.9,
            })
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    sources_event = next(i for i in items if i["type"] == "sources")
    assert sources_event["data"][0]["speaker_label"] == "Speaker 0"


@pytest.mark.asyncio
async def test_sources_event_speaker_label_none_when_absent():
    async def fake_stream(*args, **kwargs):
        yield _make_token_event("Answer.")

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_stream

    def fake_build_graph(db, llm, mode="hybrid", sources_sink=None):
        if sources_sink is not None:
            sources_sink.append({"episode_id": "e1", "start_ts": 1.0, "end_ts": 5.0, "text": "hello"})
        return mock_graph

    with patch("app.agent.streaming.build_graph", side_effect=fake_build_graph), \
         patch("app.agent.streaming.synthesise", return_value=b"bytes"), \
         patch("app.agent.streaming.ChatAnthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        items = await _collect(stream_agent_response("hi", "t1", db=MagicMock()))

    sources_event = next(i for i in items if i["type"] == "sources")
    assert sources_event["data"][0]["speaker_label"] is None
```

(This matches the exact `fake_build_graph(db, llm, mode="hybrid", sources_sink=None)` pattern already used by this file's other `sources_sink`-populating tests, e.g. `test_yields_sources_event_from_sink` and `test_sources_event_strips_extra_fields_and_caps_at_max`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v -k speaker_label`
Expected: FAIL — `KeyError: 'speaker_label'`, since the sources projection doesn't include it yet.

- [ ] **Step 3: Implement**

In `backend/app/agent/streaming.py`, find the `deduped.append({...})` block (inside the `if sources_sink:` block, building the `deduped` list) and add `speaker_label`:

```python
                    deduped.append({
                        "episode_id": src.get("episode_id"),
                        "start_ts": src.get("start_ts"),
                        "end_ts": src.get("end_ts"),
                        "text": src.get("text"),
                        "speaker_label": src.get("speaker_label"),
                    })
```

Update the module docstring's documented event shape (near the top of the file, where `{"type": "sources", "data": [{"episode_id","start_ts","end_ts","text"}, ...]}` is documented):

```
        {"type": "sources", "data": [{"episode_id","start_ts","end_ts","text","speaker_label"}, ...]}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && python -m pytest tests/agent/test_streaming.py -v`
Expected: all tests in this file pass, including the two new ones.

- [ ] **Step 5: Run the full suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass, 0 regressions.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/streaming.py backend/tests/agent/test_streaming.py
git commit -m "feat(agent): include speaker_label in streamed source citations

Deduped/capped sources sent to the client now carry speaker_label
alongside episode_id/start_ts/end_ts/text — None when diarization
never ran or the chunk's speaker was ambiguous."
```

---

### Task 7: API response models — episode summary/chapters, chunk speaker_label

**Files:**
- Modify: `backend/app/episodes/router.py`
- Modify: `backend/app/ingest/router.py`
- Test: `backend/tests/test_episodes_router.py` (create if it doesn't already exist — check first)
- Test: `backend/tests/test_ingest_router.py`

**Interfaces:**
- Consumes: `Episode.summary`, `Episode.chapters`, `Chunk.speaker_label` (Task 1) via raw SQL.
- Produces: `EpisodeSummary` (episodes/router.py, used by `GET /episodes` and `GET /episodes/{id}`) gains `summary: str | None` and `chapters: list[dict] | None`. `ChunkItem` (episodes/router.py, used by `GET /episodes/{id}/chunks`) gains `speaker_label: str | None`. `EpisodeResponse` (ingest/router.py, returned by `/ingest/upload`, `/ingest/url`, `/ingest/rss`) gains the same `summary`/`chapters` fields.

- [ ] **Step 1: Read the existing episodes router test file**

`backend/tests/test_episodes_router.py` already exists, using a `client`/`mock_db` pytest fixture pair (from `backend/tests/conftest.py` — not a manually-constructed `TestClient`), and its own `make_episode_row`/`make_chunk_row` helpers. Follow this exact convention for the new tests below — extend the two existing helpers with new optional parameters rather than writing new ones, so every existing call site (`make_episode_row()`, `make_episode_row(filename="test.mp3", chunk_count=3)`, etc.) keeps working unchanged.

- [ ] **Step 2: Write the failing tests**

In `backend/tests/test_episodes_router.py`, extend the two existing row-builder helpers:

```python
def make_episode_row(filename="ep.mp3", chunk_count=5, source_url=None, summary=None, chapters=None):
    row = MagicMock()
    row.id = uuid.uuid4()
    row.filename = filename
    row.source_url = source_url
    row.chunk_count = chunk_count
    row.summary = summary
    row.chapters = chapters
    return row


def make_chunk_row(start_ts=0.0, end_ts=10.0, text="hello", speaker_label=None):
    row = MagicMock()
    row.id = uuid.uuid4()
    row.start_ts = start_ts
    row.end_ts = end_ts
    row.text = text
    row.speaker_label = speaker_label
    return row
```

Then add:

```python
def test_get_episode_includes_summary_and_chapters(client, mock_db):
    ep = make_episode_row(
        filename="test.mp3", chunk_count=3,
        summary="A summary.", chapters=[{"start_ts": 0.0, "title": "Intro"}],
    )
    mock_db.execute.return_value.fetchone.return_value = ep
    response = client.get(f"/episodes/{ep.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "A summary."
    assert body["chapters"] == [{"start_ts": 0.0, "title": "Intro"}]


def test_get_episode_summary_and_chapters_default_to_null(client, mock_db):
    ep = make_episode_row()  # summary/chapters default to None
    mock_db.execute.return_value.fetchone.return_value = ep
    response = client.get(f"/episodes/{ep.id}")
    body = response.json()
    assert body["summary"] is None
    assert body["chapters"] is None


def test_list_episodes_includes_summary_and_chapters(client, mock_db):
    ep = make_episode_row(summary="A summary.", chapters=[{"start_ts": 0.0, "title": "Intro"}])
    mock_db.execute.return_value.fetchall.return_value = [ep]
    response = client.get("/episodes")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["summary"] == "A summary."


def test_get_episode_chunks_includes_speaker_label(client, mock_db):
    ep_id = str(uuid.uuid4())
    chunks = [make_chunk_row(speaker_label="Speaker 0")]
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{ep_id}/chunks")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["speaker_label"] == "Speaker 0"


def test_get_episode_chunks_speaker_label_null_when_absent(client, mock_db):
    ep_id = str(uuid.uuid4())
    chunks = [make_chunk_row()]  # speaker_label defaults to None
    mock_db.execute.return_value.fetchall.return_value = chunks
    response = client.get(f"/episodes/{ep_id}/chunks")
    body = response.json()
    assert body[0]["speaker_label"] is None
```

Add to `backend/tests/test_ingest_router.py`. This file's existing `make_mock_episode` helper (top of the file) doesn't set `.summary`/`.chapters` — extend it with optional parameters defaulting to `None`, so every existing call site (`make_mock_episode(chunk_count=3)` etc.) keeps working unchanged:

```python
def make_mock_episode(chunk_count=2, summary=None, chapters=None):
    ep = MagicMock()
    ep.id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    ep.filename = "test.mp3"
    ep.chunks = [MagicMock()] * chunk_count
    ep.summary = summary
    ep.chapters = chapters
    return ep
```

Then add:

```python
def test_upload_response_includes_summary_and_chapters(client):
    mock_ep = make_mock_episode(
        chunk_count=3,
        summary="A short summary.",
        chapters=[{"start_ts": 0.0, "title": "Intro"}],
    )

    with patch("app.ingest.router.ingest_audio", return_value=mock_ep):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "A short summary."
    assert body["chapters"] == [{"start_ts": 0.0, "title": "Intro"}]


def test_upload_response_summary_and_chapters_default_to_null(client):
    mock_ep = make_mock_episode(chunk_count=1)  # summary/chapters default to None

    with patch("app.ingest.router.ingest_audio", return_value=mock_ep):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.mp3", BytesIO(b"ID3fake audio"), "audio/mpeg")},
        )

    body = response.json()
    assert body["summary"] is None
    assert body["chapters"] is None
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_episodes_router.py tests/test_ingest_router.py -v -k "summary or chapters or speaker_label"`
Expected: FAIL — the response models don't have these fields yet, so the JSON responses won't include them (FastAPI/Pydantic strips unknown-to-the-model fields from ORM/dict inputs, and here we're asserting on the OUTPUT which simply won't have the keys — `body["summary"]` raises `KeyError`).

- [ ] **Step 4: Implement**

In `backend/app/episodes/router.py`:

```python
class EpisodeSummary(BaseModel):
    id: str
    filename: str
    source_url: str | None
    chunk_count: int
    summary: str | None = None
    chapters: list[dict] | None = None


class ChunkItem(BaseModel):
    id: str
    start_ts: float
    end_ts: float
    text: str
    speaker_label: str | None = None
```

Update `list_episodes`'s SQL and construction:

```python
@router.get("", response_model=list[EpisodeSummary])
def list_episodes(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT e.id, e.filename, e.source_url, e.summary, e.chapters, COUNT(c.id) AS chunk_count
        FROM episodes e
        LEFT JOIN chunks c ON c.episode_id = e.id
        GROUP BY e.id, e.filename, e.source_url, e.summary, e.chapters
        ORDER BY e.created_at DESC
    """)).fetchall()
    return [
        EpisodeSummary(
            id=str(r.id),
            filename=r.filename,
            source_url=r.source_url,
            chunk_count=r.chunk_count,
            summary=r.summary,
            chapters=r.chapters,
        )
        for r in rows
    ]
```

Update `get_episode`'s SQL and construction:

```python
@router.get("/{episode_id}", response_model=EpisodeSummary)
def get_episode(episode_id: str, db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT e.id, e.filename, e.source_url, e.summary, e.chapters, COUNT(c.id) AS chunk_count
            FROM episodes e
            LEFT JOIN chunks c ON c.episode_id = e.id
            WHERE e.id = :ep_id
            GROUP BY e.id, e.filename, e.source_url, e.summary, e.chapters
        """),
        {"ep_id": episode_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Episode not found")
    return EpisodeSummary(
        id=str(row.id),
        filename=row.filename,
        source_url=row.source_url,
        chunk_count=row.chunk_count,
        summary=row.summary,
        chapters=row.chapters,
    )
```

Update `get_episode_chunks`'s SQL and construction:

```python
@router.get("/{episode_id}/chunks", response_model=list[ChunkItem])
def get_episode_chunks(
    episode_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text("""
            SELECT id, start_ts, end_ts, text, speaker_label
            FROM chunks
            WHERE episode_id = :ep_id
            ORDER BY start_ts ASC
            LIMIT :limit
        """),
        {"ep_id": episode_id, "limit": limit},
    ).fetchall()
    return [
        ChunkItem(id=str(r.id), start_ts=r.start_ts, end_ts=r.end_ts, text=r.text, speaker_label=r.speaker_label)
        for r in rows
    ]
```

In `backend/app/ingest/router.py`:

```python
class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    summary: str | None = None
    chapters: list[dict] | None = None
```

Update all three construction sites (`/upload`, `/url`, the RSS batch loop) to pass the new fields through from the `Episode` ORM object `ingest_audio`/`ingest_from_url`/`ingest_from_rss` already returns:

```python
    return EpisodeResponse(
        id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks),
        summary=episode.summary, chapters=episode.chapters,
    )
```

(Apply this same three-field addition — `summary=episode.summary, chapters=episode.chapters` — at all three `EpisodeResponse(...)` construction sites in this file: `/upload`, `/url`, and inside the RSS batch's list comprehension using `ep` instead of `episode`.)

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && python -m pytest tests/test_episodes_router.py tests/test_ingest_router.py -v`
Expected: all pass.

- [ ] **Step 6: Run the full suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all pass, 0 regressions.

- [ ] **Step 7: Commit**

```bash
git add backend/app/episodes/router.py backend/app/ingest/router.py backend/tests/test_episodes_router.py backend/tests/test_ingest_router.py
git commit -m "feat(api): expose summary/chapters/speaker_label on episode + chunk endpoints

EpisodeSummary (GET /episodes, GET /episodes/{id}), EpisodeResponse
(ingest completion responses), and ChunkItem (GET /episodes/{id}/chunks)
all carry the new Etap 16 fields, defaulting to None until an episode
has actually been through diarization/summarization."
```

---

### Task 8: Frontend — `SourcePlayer.tsx` prefixes citations with the speaker label

**Files:**
- Modify: `frontend/components/SourcePlayer.tsx`
- Modify: `frontend/components/ChatUI.tsx` (the `Source` interface only)
- Test: `frontend/components/__tests__/SourcePlayer.test.tsx` (create — check first whether a test file for this component already exists)

**Interfaces:**
- Consumes: `speaker_label: string | null` field now present on the `sources` array the backend sends (Task 6).
- Produces: `SourcePlayer`'s citation card shows `"Speaker 0: hello there..."` (bold/colored speaker prefix, then the citation text) when `speaker_label` is present, and the existing unprefixed text when it's `null`/absent.

- [ ] **Step 1: Check for an existing test file**

Run: `ls frontend/components/__tests__/ | grep -i sourceplayer`. If it exists, read it and follow its conventions. This plan assumes it does not yet exist and creates one.

- [ ] **Step 2: Write the failing tests**

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SourcePlayer from "../SourcePlayer";

describe("SourcePlayer speaker labels", () => {
  it("prefixes the citation text with the speaker label when present", () => {
    render(
      <SourcePlayer
        sources={[
          { episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there", speaker_label: "Speaker 0" },
        ]}
      />
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });

  it("renders unprefixed citation text when speaker_label is null", () => {
    render(
      <SourcePlayer
        sources={[{ episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there", speaker_label: null }]}
      />
    );
    expect(screen.queryByText(/^Speaker \d/)).not.toBeInTheDocument();
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });

  it("renders unprefixed citation text when speaker_label is entirely absent", () => {
    render(
      <SourcePlayer sources={[{ episode_id: "e1", start_ts: 1.0, end_ts: 5.0, text: "hello there" }]} />
    );
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd frontend && npm test -- SourcePlayer`
Expected: FAIL — TypeScript compile error or a passing-but-not-actually-checking-anything state, since `Source` doesn't have `speaker_label` and nothing renders it. If TypeScript strictness makes this a compile error rather than a clean runtime failure, that's expected and fine — the next step's interface change resolves it.

- [ ] **Step 4: Implement**

In `frontend/components/SourcePlayer.tsx`, add the field to the local `Source` interface and render it:

```typescript
interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
  speaker_label?: string | null;
}
```

In the card's text rendering (replace the existing `<p className="text-gray-300 line-clamp-3">{src.text}</p>` line):

```tsx
            <p className="text-gray-300 line-clamp-3">
              {src.speaker_label && (
                <span className="text-indigo-300 font-semibold">{src.speaker_label}: </span>
              )}
              {src.text}
            </p>
```

In `frontend/components/ChatUI.tsx`, add the same optional field to the exported `Source` interface (used by `applyStreamEvent` and passed down to `SourcePlayer` — TypeScript structural typing means this doesn't need to literally match `SourcePlayer`'s own interface name, just its shape):

```typescript
export interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
  speaker_label?: string | null;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd frontend && npm test -- SourcePlayer`
Expected: all 3 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `cd frontend && npm test`
Expected: all pass, 0 regressions.

- [ ] **Step 7: Commit**

```bash
git add frontend/components/SourcePlayer.tsx frontend/components/ChatUI.tsx frontend/components/__tests__/SourcePlayer.test.tsx
git commit -m "feat(frontend): SourcePlayer prefixes citations with the speaker label

Falls back to today's unprefixed rendering when speaker_label is
null/absent — every episode ingested before this etap, or diarized
without a confident speaker match, keeps working exactly as before."
```

---

### Task 9: Frontend — episode summary + clickable chapters in the sidebar

**Files:**
- Modify: `frontend/components/FileUploader.tsx`
- Test: `frontend/components/__tests__/FileUploader.test.tsx` (create — check first whether one already exists)

**Interfaces:**
- Consumes: `Episode.summary: string | null`, `Episode.chapters: {start_ts: number; title: string}[] | null` (Task 7's `EpisodeSummary` API model, fetched via the existing `GET /api/episodes` call in `frontend/app/page.tsx`).
- Produces: when an episode is `selected` (focused) in the sidebar list, an expanded panel renders below its row showing the summary text and a clickable chapter list; clicking a chapter jumps an inline `<audio>` element (rendered only when `source_url` is present, matching `SourcePlayer.tsx`'s existing pattern for uploaded-file episodes with no URL) to that timestamp.

- [ ] **Step 1: Check for an existing test file**

Run: `ls frontend/components/__tests__/ | grep -i fileuploader`. If one exists, read it first and follow its conventions/mocking setup exactly (especially how it mocks `fetch` for the upload/RSS flows, if at all — the new tests below only touch the episode-list rendering, not the upload flows, so they likely don't need to mock `fetch`).

- [ ] **Step 2: Write the failing tests**

```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import FileUploader, { Episode } from "../FileUploader";

const baseEpisode: Episode = {
  id: "e1",
  filename: "test.mp3",
  chunk_count: 5,
  source_url: "https://example.com/audio.mp3",
  summary: "A great episode about testing.",
  chapters: [
    { start_ts: 0, title: "Introduction" },
    { start_ts: 120, title: "Deep dive" },
  ],
};

interface RenderOverrides {
  selectedEpisodeId?: string | null;
  episodes?: Episode[];
}

function renderUploader(overrides: RenderOverrides = {}) {
  return render(
    <FileUploader
      selectedEpisodeId={overrides.selectedEpisodeId ?? null}
      onSelectEpisode={vi.fn()}
      episodes={overrides.episodes ?? [baseEpisode]}
      episodesLoaded={true}
      onIngested={vi.fn()}
    />
  );
}

describe("FileUploader focused-episode summary/chapters panel", () => {
  it("does not show the summary panel when the episode is not selected", () => {
    renderUploader({ selectedEpisodeId: null });
    expect(screen.queryByText(/A great episode about testing/)).not.toBeInTheDocument();
  });

  it("shows the summary and chapter list when the episode is selected", () => {
    renderUploader({ selectedEpisodeId: "e1" });
    expect(screen.getByText(/A great episode about testing/)).toBeInTheDocument();
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Deep dive")).toBeInTheDocument();
  });

  it("does not render a chapters section when the episode has no chapters", () => {
    const noChapters = { ...baseEpisode, chapters: null };
    renderUploader({ selectedEpisodeId: "e1", episodes: [noChapters] });
    expect(screen.queryByText("Introduction")).not.toBeInTheDocument();
  });

  it("clicking a chapter sets the inline audio player's time via a #t= fragment", async () => {
    const user = userEvent.setup();
    renderUploader({ selectedEpisodeId: "e1" });

    const chapterButton = screen.getByRole("button", { name: /Deep dive/ });
    await user.click(chapterButton);

    const audio = screen.getByTestId("episode-audio-player") as HTMLAudioElement;
    expect(audio.src).toContain("#t=120");
  });

  it("does not render an audio player when the episode has no source_url", () => {
    const noUrl = { ...baseEpisode, source_url: null };
    renderUploader({ selectedEpisodeId: "e1", episodes: [noUrl] });
    expect(screen.queryByTestId("episode-audio-player")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd frontend && npm test -- FileUploader`
Expected: FAIL — `Episode` interface doesn't have `summary`/`chapters` yet (TypeScript error), and no expanded panel exists in the component yet.

- [ ] **Step 4: Implement**

In `frontend/components/FileUploader.tsx`, extend the `Episode` interface:

```typescript
export interface Episode {
  id: string;
  filename: string;
  chunk_count: number;
  source_url?: string | null;
  summary?: string | null;
  chapters?: { start_ts: number; title: string }[] | null;
}
```

Add state for the currently-jumped-to chapter timestamp, near the component's other `useState` calls:

```typescript
  const [chapterJumpTs, setChapterJumpTs] = useState<number | null>(null);
```

Add a small time-formatting helper near the top of the file (module scope, outside the component — mirrors `SourcePlayer.tsx`'s own `formatTime`, but this file doesn't import from `SourcePlayer.tsx` so it needs its own copy rather than a cross-component import, keeping the two components independently understandable):

```typescript
function formatChapterTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
```

In the episode list's `.map((ep) => ...)` block, after the existing closing `</button>` for each episode row (still inside the `.map()`'s returned fragment — you'll need to wrap the existing `<button>` and this new panel in a fragment or a `<div key={ep.id}>` instead of the button being the top-level returned element), add the expanded panel, rendered only `{selected && (...)}`:

```tsx
          episodes.map((ep) => {
            const selected = selectedEpisodeId === ep.id;
            return (
              <div key={ep.id}>
                <button
                  onClick={() => toggleEpisode(ep)}
                  aria-pressed={selected}
                  title={selected ? "Click to deselect" : "Click to focus queries on this episode"}
                  className={`text-left w-full rounded-lg border px-3 py-2 transition-colors group ${
                    selected
                      ? "bg-indigo-950/60 border-indigo-600/70"
                      : "bg-gray-900 border-gray-800 hover:border-gray-600"
                  }`}
                >
                  <div className="flex items-start justify-between gap-1">
                    <p
                      className={`text-xs font-medium truncate ${
                        selected ? "text-indigo-200" : "text-gray-300"
                      }`}
                      title={ep.filename}
                    >
                      {ep.filename}
                    </p>
                    {selected && (
                      <span className="shrink-0 text-indigo-400 text-[10px] mt-0.5">
                        <span aria-hidden="true">✓</span> focused
                      </span>
                    )}
                  </div>
                  <p className={`text-[10px] mt-0.5 ${selected ? "text-indigo-400/70" : "text-gray-600"}`}>
                    {ep.chunk_count} chunks
                  </p>
                </button>
                {selected && (ep.summary || ep.chapters?.length || ep.source_url) && (
                  <div className="mt-1 mb-2 px-3 py-2 bg-gray-950/60 border border-indigo-900/40 rounded-lg text-xs space-y-2">
                    {ep.summary && <p className="text-gray-300">{ep.summary}</p>}
                    {ep.chapters && ep.chapters.length > 0 && (
                      <div className="space-y-1">
                        <p className="text-[10px] uppercase tracking-wider text-gray-500">Chapters</p>
                        {ep.chapters.map((ch, i) => (
                          <button
                            key={i}
                            onClick={() => setChapterJumpTs(ch.start_ts)}
                            className="flex items-center gap-2 w-full text-left hover:bg-gray-800/60 rounded px-1 py-0.5"
                          >
                            <span className="text-indigo-400 font-mono text-[10px] shrink-0">
                              {formatChapterTime(ch.start_ts)}
                            </span>
                            <span className="text-gray-300 truncate">{ch.title}</span>
                          </button>
                        ))}
                      </div>
                    )}
                    {ep.source_url && (
                      <audio
                        data-testid="episode-audio-player"
                        controls
                        preload="none"
                        src={
                          chapterJumpTs !== null
                            ? `${ep.source_url.split("#")[0]}#t=${chapterJumpTs}`
                            : ep.source_url
                        }
                        className="w-full h-8"
                      >
                        Your browser does not support audio playback.
                      </audio>
                    )}
                  </div>
                )}
              </div>
            );
          })
```

Note: `chapterJumpTs` is intentionally a single piece of state shared across whichever episode is currently selected (only one episode can be `selected`/focused at a time, per the existing `toggleEpisode` logic, so this doesn't need to be keyed per-episode).

- [ ] **Step 5: Run to verify it passes**

Run: `cd frontend && npm test -- FileUploader`
Expected: all 5 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `cd frontend && npm test`
Expected: all pass, 0 regressions. Also run `cd frontend && npx tsc --noEmit` to confirm no type errors — `FileUploader.tsx`'s `Episode` export is imported by `ChatUI.tsx` and `SourcePlayer.tsx`, so confirm neither breaks from the new optional fields (they shouldn't, since both are optional and additive).

- [ ] **Step 7: Commit**

```bash
git add frontend/components/FileUploader.tsx frontend/components/__tests__/FileUploader.test.tsx
git commit -m "feat(frontend): episode summary + clickable chapter list in the sidebar

Renders inline when an episode is focused, reusing the existing
audio-with-#t=-fragment jump pattern SourcePlayer.tsx already
established for citation timestamps. No-op (nothing renders) for
episodes ingested before this etap or without a source_url."
```

---

### Task 10: Live verification, env docs, DECISIONS.md

**Files:**
- Modify: `CLAUDE.md` (new columns, speaker-mapping algorithm, structured-output contract, where summary/chapters appear in the UI — per the spec's own "Do CLAUDE.md" list)
- Modify: `DECISIONS.md`
- No test file — this task's deliverable is manual verification evidence + documentation, same shape as Etap 15's Task 10.

- [ ] **Step 1: Add the CLAUDE.md documentation (safe, no live key needed)**

Add a short section to `CLAUDE.md` (near the existing "Critical data shape — chunk record" section, since this directly extends that invariant) documenting:
- The new `Chunk.speaker_label` (nullable) and `Episode.summary`/`Episode.chapters` (nullable) columns, and that they follow the same idempotent-`ALTER TABLE`-in-`init_db()` pattern as `text_search`.
- The speaker→chunk mapping algorithm in one sentence: majority time-overlap with a >50% threshold, `NULL` below that — never guessed.
- The summarizer's structured-output contract: one Claude Haiku tool-use call producing `{summary, chapters, speaker_names}`, `speaker_names` best-effort remapping `"Speaker N"` onto a real name only when the transcript actually states it.
- Where this shows up in the UI: `SourcePlayer.tsx` prefixes citations with the speaker label; the sidebar's focused-episode panel (`FileUploader.tsx`) shows the summary and a clickable chapter list.

Add `DEEPGRAM_API_KEY` is already documented (Etap 15) — no new env var for this etap (same account, same key, prerecorded API instead of streaming). Confirm this by checking `.env.example`/`CLAUDE.md`'s existing `DEEPGRAM_API_KEY` row's description still reads sensibly for both use cases, or tweak its one-line description to mention both (streaming voice AND diarization) if it currently only mentions Etap 15's use.

- [ ] **Step 2: Commit the safe documentation**

```bash
git add CLAUDE.md .env.example
git commit -m "docs(etap-16): document speaker_label/summary/chapters schema and contracts"
```

- [ ] **Step 3: Manual verification — needs a real DEEPGRAM_API_KEY and ANTHROPIC_API_KEY, and genuine multi-speaker audio**

This step cannot be performed by an automated agent in this session — it requires a real `DEEPGRAM_API_KEY` (confirm one is configured; Etap 15 already required this key, so it may already be set from that etap's live verification), a short (~5-10 min) multi-speaker podcast clip, and a human listening to verify the diarization output is actually correct (the DoD explicitly requires ear-verification: "zweryfikowane ręcznie ze słuchem").

1. Run `docker compose up`, ingest a short multi-speaker podcast episode (via file upload, URL, or RSS).
2. Confirm the diarization + summarization live wire shape matches what `diarizer.py`/`summarizer.py` assume — per Task 2's explicit warning, this is the same category of risk that has bitten this project twice before. If Deepgram's real prerecorded-diarization response shape differs from what `diarize_audio` parses, fix the parsing and add a regression test using the real captured shape.
3. Open the episode in the UI, ask a question whose answer cites a specific speaker's segment — confirm the citation shows a `Speaker N` (or real name, if the summarizer's `speaker_names` mapping caught one) prefix, and listen to that timestamp range to confirm the label is actually correct for that speaker (not swapped/misattributed).
4. Confirm the episode's sidebar panel shows a summary and a clickable chapter list; click a chapter and confirm the inline audio player jumps to roughly that point.
5. Simulate a diarization/summarization failure (temporarily set `DEEPGRAM_API_KEY`/`ANTHROPIC_API_KEY` to an invalid value, or unset one), ingest another episode, and confirm: the episode still ingests successfully, it's still fully searchable (ask a question about it, confirm the agent can answer), and its citations/sidebar panel simply show no speaker label / no summary — no error surfaced to the user, no crash. Restore the real keys afterward.

- [ ] **Step 4: Write the `DECISIONS.md` entry**

Follow the existing entry format in the file (`## [Etap 16] <title> (<date>)`, Decyzja/Alternatywy odrzucone/Uzasadnienie structure — see any recent `## [Etap 15] ...` entry for the exact shape). Cover: why >50% overlap threshold (not majority-of-words or some other heuristic) was chosen for speaker assignment, what the live verification in Step 3 actually found (did Deepgram's real response shape match the assumed one? did diarization accuracy hold up on ear-check?), and the graceful-degradation behavior confirmed in Step 3.5.

- [ ] **Step 5: Commit**

```bash
git add DECISIONS.md
git commit -m "docs(etap-16): live verification results and speaker-mapping threshold rationale"
```

---

## Self-Review Notes

- **Spec coverage:** DB schema (Task 1), `diarizer.py` with `diarize=true` + overlap mapping (Task 2), `summarizer.py` with one structured Claude Haiku call including best-effort speaker names (Task 3), integration into `ingest_audio()` for all three ingest paths since they all funnel through it (Task 4), `GET /episodes/{id}` and list/chunks responses extended (Task 7), `SourcePlayer.tsx` speaker prefix + sidebar summary/chapters (Tasks 8-9) — all five "Zakres" bullets covered. DoD's four requirements map to Task 10's manual verification (items 1, 3 — diarization accuracy and graceful degradation) and Tasks 8-9's automated tests (item 2 — summary/chapters render and jump correctly, verified by component tests plus Task 10's manual click-through).
- **Out-of-scope items honored:** no voice fingerprinting across episodes, no manual speaker-label editing UI — neither appears in any task.
- **Type/name consistency checked:** `diarize_audio`/`assign_speaker_labels`'s exact signatures (Task 2) match how Task 4's `service.py` calls them; `summarize_episode`'s return shape `{summary, chapters, speaker_names}` (Task 3) matches exactly how Task 4 destructures it; `speaker_label`/`summary`/`chapters` field names are identical across the ORM columns (Task 1), search dicts (Task 5), streaming sources (Task 6), API response models (Task 7), and both frontend `Source`/`Episode` interfaces (Tasks 8-9) — no `speakerLabel` vs `speaker_label` mismatches, no `chapter_list` vs `chapters` mismatches.
- **Pułapka (spec's own trap warning) addressed:** the >50%-of-chunk-duration threshold is implemented exactly as specified in Task 2's `assign_speaker_labels`, with an explicit boundary test (`test_assign_speaker_labels_returns_none_below_threshold`, asserting exactly-50% returns `None`, not just "high" vs "low"). Structured output is enforced via Anthropic tool-use/`tool_choice`, never regex — Task 3's `summarize_episode` raises rather than falling back to text-parsing if the model doesn't return a tool-use block.
