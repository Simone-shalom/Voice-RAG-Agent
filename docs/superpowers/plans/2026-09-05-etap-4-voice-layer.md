# Etap 4 — Voice Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add voice I/O to the system: browser records a question (or types it), backend transcribes via Whisper STT, agent answers, ElevenLabs TTS speaks the reply, and a source player lets users jump to any cited timestamp in the original recording.

**Architecture:** Two new backend endpoints — `POST /voice/stt` (audio → text via Whisper) and `POST /voice/tts` (text → MP3 via ElevenLabs). The existing `POST /agent/chat` endpoint handles the agent turn. A minimal Next.js 15 App Router frontend wires them together: `AudioRecorder` captures mic input, `ChatUI` handles text input and reply display, `SourcePlayer` plays the original episode audio and jumps to cited timestamps. No streaming LLM for Etap 4 (streaming added in a follow-up if needed). Time-to-first-audio recorded in DECISIONS.md.

**Simplification applied (per CLAUDE.md):** Mic recording is optional UX; text input is the fallback path and is always present. TTS playback is the primary "voice output" — that's what the DoD tests require.

**Tech Stack:** Python (ElevenLabs SDK, OpenAI Whisper), Next.js 15 (App Router), TypeScript, Tailwind CSS (via CDN in plain HTML fallback if needed).

---

## File Map

**Create (backend):**
- `backend/app/voice/__init__.py`
- `backend/app/voice/stt.py` — `transcribe_audio(file_bytes, filename) -> str`
- `backend/app/voice/tts.py` — `synthesise(text) -> bytes` (MP3)
- `backend/app/voice/router.py` — `POST /voice/stt`, `POST /voice/tts`
- `backend/tests/test_voice_stt.py`
- `backend/tests/test_voice_tts.py`
- `backend/tests/test_voice_router.py`

**Modify (backend):**
- `backend/requirements.txt` — add `elevenlabs`
- `backend/app/main.py` — include voice router

**Create (frontend):**
- `frontend/package.json`
- `frontend/tsconfig.json`
- `frontend/next.config.ts`
- `frontend/app/layout.tsx`
- `frontend/app/page.tsx`
- `frontend/app/globals.css`
- `frontend/components/ChatUI.tsx`
- `frontend/components/AudioRecorder.tsx`
- `frontend/components/SourcePlayer.tsx`

---

## Task 1: Backend voice STT endpoint

**Files:**
- Create: `backend/app/voice/__init__.py`
- Create: `backend/app/voice/stt.py`
- Create: `backend/tests/test_voice_stt.py`

`transcribe_audio` wraps the same Whisper API call as the ingest transcriber but returns a single string (the full text) rather than segmented dicts — simpler output for a question transcription.

- [ ] **Step 1: Write failing tests**

`backend/tests/test_voice_stt.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.voice.stt import transcribe_audio


def make_mock_client(text="hello world"):
    mock = MagicMock()
    mock.audio.transcriptions.create.return_value = MagicMock(text=text)
    return mock


def test_transcribe_audio_returns_string():
    with patch("app.voice.stt._client", return_value=make_mock_client("What is AI?")):
        result = transcribe_audio(b"fake audio bytes", "question.webm")

    assert isinstance(result, str)
    assert result == "What is AI?"


def test_transcribe_audio_passes_file_bytes():
    mock_client = make_mock_client()
    with patch("app.voice.stt._client", return_value=mock_client):
        transcribe_audio(b"audio", "q.mp3")

    mock_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "whisper-1"


def test_transcribe_audio_empty_bytes_raises():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        transcribe_audio(b"", "q.webm")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_voice_stt.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.voice'`

- [ ] **Step 3: Create `backend/app/voice/__init__.py`** (empty)

- [ ] **Step 4: Create `backend/app/voice/stt.py`**

```python
import io
import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio bytes to text using Whisper API.
    Returns the full transcript as a plain string.
    Raises ValueError for empty input.
    """
    if not file_bytes:
        raise ValueError("audio bytes are empty")
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = filename
    response = _client().audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return response.text
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_voice_stt.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/voice/__init__.py backend/app/voice/stt.py backend/tests/test_voice_stt.py
git commit -m "feat(etap-4): add Whisper STT transcription for voice input"
```

---

## Task 2: Backend voice TTS endpoint

**Files:**
- Create: `backend/app/voice/tts.py`
- Create: `backend/tests/test_voice_tts.py`
- Modify: `backend/requirements.txt`

ElevenLabs SDK lazy client — same `@lru_cache` pattern as other API clients. Missing `ELEVENLABS_API_KEY` raises `RuntimeError`. Returns raw MP3 bytes; the router streams them with `Content-Type: audio/mpeg`.

- [ ] **Step 1: Add `elevenlabs` to requirements and install**

Append to `backend/requirements.txt`:
```
elevenlabs==1.50.5
```

```bash
cd backend && .venv/Scripts/pip.exe install elevenlabs==1.50.5
```
If version conflict, install with `elevenlabs>=1` and pin the resolved version.

- [ ] **Step 2: Write failing tests**

`backend/tests/test_voice_tts.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
import pytest
from app.voice.tts import synthesise


def make_mock_client(audio_bytes=b"fake_mp3"):
    mock = MagicMock()
    mock.text_to_speech.convert.return_value = iter([audio_bytes])
    return mock


def test_synthesise_returns_bytes():
    with patch("app.voice.tts._client", return_value=make_mock_client(b"mp3data")):
        result = synthesise("Hello world")

    assert isinstance(result, bytes)
    assert result == b"mp3data"


def test_synthesise_raises_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        from app.voice import tts
        tts._client.cache_clear()
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
            synthesise("test")
    tts._client.cache_clear()


def test_synthesise_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        synthesise("")
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_voice_tts.py -v
```
Expected: import error.

- [ ] **Step 4: Create `backend/app/voice/tts.py`**

```python
import os
from functools import lru_cache

from elevenlabs import ElevenLabs


VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George" — change in .env if desired


@lru_cache(maxsize=1)
def _client() -> ElevenLabs:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set — cannot use TTS")
    return ElevenLabs(api_key=api_key)


def synthesise(text: str, voice_id: str = VOICE_ID) -> bytes:
    """
    Convert text to MP3 audio using ElevenLabs TTS.
    Returns raw MP3 bytes.
    Raises ValueError for empty text, RuntimeError if API key unset.
    """
    if not text:
        raise ValueError("text is empty")
    chunks = _client().text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
    )
    return b"".join(chunks)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_voice_tts.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/voice/tts.py backend/tests/test_voice_tts.py
git commit -m "feat(etap-4): add ElevenLabs TTS synthesis with lazy client"
```

---

## Task 3: Voice FastAPI router

**Files:**
- Create: `backend/app/voice/router.py`
- Create: `backend/tests/test_voice_router.py`
- Modify: `backend/app/main.py`

Two endpoints:
- `POST /voice/stt` — multipart form upload, returns `{text: str}`
- `POST /voice/tts` — JSON body `{text: str}`, returns MP3 stream

- [ ] **Step 1: Write failing tests**

`backend/tests/test_voice_router.py`:
```python
import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")

from unittest.mock import patch


def test_stt_returns_text(client):
    with patch("app.voice.router.transcribe_audio", return_value="What is AI?"):
        response = client.post(
            "/voice/stt",
            files={"audio": ("q.webm", b"fake audio", "audio/webm")},
        )
    assert response.status_code == 200
    assert response.json()["text"] == "What is AI?"


def test_stt_empty_file_returns_400(client):
    with patch("app.voice.router.transcribe_audio", side_effect=ValueError("audio bytes are empty")):
        response = client.post(
            "/voice/stt",
            files={"audio": ("q.webm", b"", "audio/webm")},
        )
    assert response.status_code == 400


def test_tts_returns_audio(client):
    with patch("app.voice.router.synthesise", return_value=b"mp3bytes"):
        response = client.post("/voice/tts", json={"text": "Hello world"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"mp3bytes"


def test_tts_empty_text_returns_400(client):
    with patch("app.voice.router.synthesise", side_effect=ValueError("text is empty")):
        response = client.post("/voice/tts", json={"text": ""})
    assert response.status_code == 400


def test_tts_missing_key_returns_400(client):
    with patch("app.voice.router.synthesise", side_effect=RuntimeError("ELEVENLABS_API_KEY not set")):
        response = client.post("/voice/tts", json={"text": "hello"})
    assert response.status_code == 400
    assert "ELEVENLABS_API_KEY" in response.json()["detail"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/test_voice_router.py -v
```
Expected: import error.

- [ ] **Step 3: Create `backend/app/voice/router.py`**

```python
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .stt import transcribe_audio
from .tts import synthesise

router = APIRouter(prefix="/voice", tags=["voice"])


class TTSRequest(BaseModel):
    text: str


class STTResponse(BaseModel):
    text: str


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        file_bytes = await audio.read()
        text = transcribe_audio(file_bytes, audio.filename or "audio.webm")
        return STTResponse(text=text)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        audio_bytes = synthesise(request.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Update `backend/app/main.py`** to add voice router

```python
from .voice.router import router as voice_router
# add after agent_router line:
app.include_router(voice_router)
```

- [ ] **Step 5: Run all tests to confirm they pass**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 87 passed (77 + 10 new).

- [ ] **Step 6: Commit**

```bash
git add backend/app/voice/router.py backend/app/main.py backend/tests/test_voice_router.py
git commit -m "feat(etap-4): add POST /voice/stt and POST /voice/tts endpoints"
```

---

## Task 4: Next.js frontend scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/.env.local.example`

Next.js 15 with App Router, TypeScript, Tailwind CSS.

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "voice-rag-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "15.1.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/node": "^22",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "typescript": "^5",
    "tailwindcss": "^3",
    "autoprefixer": "^10",
    "postcss": "^8"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Create `frontend/next.config.ts`**

```ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 4: Create `frontend/postcss.config.js`**

```js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 5: Create `frontend/tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 6: Create `frontend/.env.local.example`**

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

- [ ] **Step 7: Install dependencies**

```bash
cd frontend && npm install
```
Expected: node_modules created.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "chore(etap-4): scaffold Next.js 15 frontend with Tailwind CSS"
```

---

## Task 5: Frontend components and page

**Files:**
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/globals.css`
- Create: `frontend/app/page.tsx`
- Create: `frontend/components/ChatUI.tsx`
- Create: `frontend/components/AudioRecorder.tsx`
- Create: `frontend/components/SourcePlayer.tsx`

- [ ] **Step 1: Create `frontend/app/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 2: Create `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Knowledge Agent",
  description: "Ask questions about your podcast library",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen">{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Create `frontend/components/SourcePlayer.tsx`**

```tsx
"use client";
import { useRef, useEffect } from "react";

interface Props {
  episodeUrl: string;
  jumpToSeconds: number | null;
}

export default function SourcePlayer({ episodeUrl, jumpToSeconds }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (jumpToSeconds !== null && audioRef.current) {
      audioRef.current.currentTime = jumpToSeconds;
      audioRef.current.play().catch(() => {});
    }
  }, [jumpToSeconds]);

  return (
    <div className="mt-4">
      <p className="text-sm text-gray-500 mb-1">Source audio</p>
      <audio ref={audioRef} controls src={episodeUrl} className="w-full" />
    </div>
  );
}
```

- [ ] **Step 4: Create `frontend/components/AudioRecorder.tsx`**

```tsx
"use client";
import { useState, useRef } from "react";

interface Props {
  onTranscription: (text: string) => void;
  backendUrl: string;
}

export default function AudioRecorder({ onTranscription, backendUrl }: Props) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    chunksRef.current = [];
    mr.ondataavailable = (e) => chunksRef.current.push(e.data);
    mr.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const form = new FormData();
      form.append("audio", blob, "question.webm");
      const res = await fetch(`${backendUrl}/voice/stt`, { method: "POST", body: form });
      const data = await res.json();
      onTranscription(data.text ?? "");
      stream.getTracks().forEach((t) => t.stop());
    };
    mediaRecorderRef.current = mr;
    mr.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  return (
    <button
      onMouseDown={startRecording}
      onMouseUp={stopRecording}
      className={`px-4 py-2 rounded text-white font-semibold ${
        recording ? "bg-red-500" : "bg-blue-500 hover:bg-blue-600"
      }`}
    >
      {recording ? "● Recording…" : "🎙 Hold to speak"}
    </button>
  );
}
```

- [ ] **Step 5: Create `frontend/components/ChatUI.tsx`**

```tsx
"use client";
import { useState } from "react";

interface Chunk {
  chunk_id: string;
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
  similarity: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  chunks?: Chunk[];
  audioUrl?: string;
}

interface Props {
  backendUrl: string;
  onCitationClick: (episodeId: string, timestamp: number) => void;
  threadId: string;
  initialText?: string;
  onInitialTextConsumed?: () => void;
}

export default function ChatUI({
  backendUrl,
  onCitationClick,
  threadId,
  initialText,
  onInitialTextConsumed,
}: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState(initialText ?? "");
  const [loading, setLoading] = useState(false);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);
    onInitialTextConsumed?.();

    try {
      const agentRes = await fetch(`${backendUrl}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, thread_id: threadId }),
      });
      const agentData = await agentRes.json();
      const reply: string = agentData.reply ?? "";

      // Get TTS audio
      let audioUrl: string | undefined;
      try {
        const ttsRes = await fetch(`${backendUrl}/voice/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reply }),
        });
        if (ttsRes.ok) {
          const blob = await ttsRes.blob();
          audioUrl = URL.createObjectURL(blob);
        }
      } catch {}

      // Parse citations from reply (timestamps like "12.3s" or "at 5.2s")
      const chunks: Chunk[] = [];
      const tsMatches = reply.matchAll(/(\d+(?:\.\d+)?)s/g);
      for (const m of tsMatches) {
        chunks.push({
          chunk_id: m[0],
          episode_id: "ep-unknown",
          start_ts: parseFloat(m[1]),
          end_ts: parseFloat(m[1]) + 5,
          text: `Cited at ${m[0]}`,
          similarity: 1.0,
        });
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: reply, chunks, audioUrl },
      ]);
      if (audioUrl) {
        new Audio(audioUrl).play().catch(() => {});
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 min-h-64 max-h-96 overflow-y-auto">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              msg.role === "user"
                ? "bg-blue-100 self-end max-w-prose"
                : "bg-white border self-start max-w-prose"
            }`}
          >
            <p className="text-sm">{msg.content}</p>
            {msg.chunks && msg.chunks.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {msg.chunks.map((c, j) => (
                  <button
                    key={j}
                    onClick={() => onCitationClick(c.episode_id, c.start_ts)}
                    className="text-xs bg-yellow-100 hover:bg-yellow-200 border border-yellow-300 rounded px-2 py-0.5"
                  >
                    ▶ {c.start_ts.toFixed(1)}s
                  </button>
                ))}
              </div>
            )}
            {msg.audioUrl && (
              <audio controls src={msg.audioUrl} className="mt-2 w-full h-8" />
            )}
          </div>
        ))}
        {loading && (
          <div className="self-start text-gray-400 text-sm animate-pulse">Thinking…</div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          className="flex-1 border rounded px-3 py-2 text-sm"
          placeholder="Ask about the audio library…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage(input)}
        />
        <button
          onClick={() => sendMessage(input)}
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Create `frontend/app/page.tsx`**

```tsx
"use client";
import { useState, useCallback } from "react";
import ChatUI from "@/components/ChatUI";
import AudioRecorder from "@/components/AudioRecorder";
import SourcePlayer from "@/components/SourcePlayer";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const THREAD_ID = `session-${Math.random().toString(36).slice(2, 8)}`;

export default function Home() {
  const [episodeUrl, setEpisodeUrl] = useState<string>("");
  const [jumpTo, setJumpTo] = useState<number | null>(null);
  const [transcribedText, setTranscribedText] = useState<string>("");

  const handleCitationClick = useCallback((episodeId: string, timestamp: number) => {
    setJumpTo(timestamp);
  }, []);

  return (
    <main className="max-w-2xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-2">Voice Knowledge Agent</h1>
      <p className="text-gray-500 mb-6 text-sm">
        Ask questions about your audio library by voice or text.
      </p>

      <section className="bg-white border rounded-xl p-4 mb-4 shadow-sm">
        <h2 className="text-sm font-semibold text-gray-600 mb-3">Ask a question</h2>
        <div className="mb-3">
          <AudioRecorder
            onTranscription={setTranscribedText}
            backendUrl={BACKEND_URL}
          />
        </div>
        <ChatUI
          backendUrl={BACKEND_URL}
          onCitationClick={handleCitationClick}
          threadId={THREAD_ID}
          initialText={transcribedText}
          onInitialTextConsumed={() => setTranscribedText("")}
        />
      </section>

      <section className="bg-white border rounded-xl p-4 shadow-sm">
        <h2 className="text-sm font-semibold text-gray-600 mb-2">Source audio player</h2>
        <input
          className="w-full border rounded px-3 py-2 text-sm mb-2"
          placeholder="Paste episode URL or local path to hear the source recording"
          value={episodeUrl}
          onChange={(e) => setEpisodeUrl(e.target.value)}
        />
        {episodeUrl && (
          <SourcePlayer episodeUrl={episodeUrl} jumpToSeconds={jumpTo} />
        )}
      </section>
    </main>
  );
}
```

- [ ] **Step 7: Run `next build` to verify no type errors**

```bash
cd frontend && npm run build 2>&1 | tail -20
```
Expected: `✓ Compiled successfully` or `Route (app) Size`.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat(etap-4): add Next.js frontend with ChatUI, AudioRecorder, SourcePlayer"
```

---

## Task 6: Full test suite verification

- [ ] **Step 1: Run all backend tests**

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 87 passed (77 baseline + 10 new voice tests).

- [ ] **Step 2: Record DECISIONS.md note on latency**

After seeing `/voice/tts` response time in browser devtools (or estimating from ElevenLabs latency ~1-1.5s + LLM inference ~1-2s), write the estimate into DECISIONS.md.

Target: time-to-first-audio < 4s (Whisper STT ~1s + agent ~1.5s + TTS ~1.5s).

---

## Self-Review

**Spec coverage:**
- [x] STT: Whisper via `/voice/stt` → Task 1
- [x] TTS: ElevenLabs via `/voice/tts` → Task 2
- [x] Mic recording: AudioRecorder component → Task 5
- [x] Agent processes → existing `/agent/chat` → Task 3 (wired in ChatUI)
- [x] Source player with timestamp jump → SourcePlayer component → Task 5
- [x] Citation click → timestamp jump → Task 5 (onCitationClick in ChatUI)
- [x] Latency target in DECISIONS.md → Task 6

**Placeholder scan:** None — all tasks have complete code.

**Type consistency:**
- `transcribe_audio(bytes, str) -> str` → used in router `transcribe_audio(file_bytes, filename)` ✓
- `synthesise(str) -> bytes` → used in router `synthesise(request.text)` ✓
- `onCitationClick(episodeId, timestamp)` in ChatUI → matches `handleCitationClick(episodeId, timestamp)` in page.tsx ✓
- `SourcePlayer({ episodeUrl, jumpToSeconds })` props → passed from page.tsx ✓
