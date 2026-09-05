# Etap 4: Voice Layer

## What was added

**Backend (`backend/app/voice/`):**
- `stt.py` — Whisper transcription via OpenAI client. Returns plain `str`. Raises `ValueError` for empty input.
- `tts.py` — ElevenLabs TTS via direct `httpx.post` (SDK avoided; see DECISIONS.md). Returns raw MP3 bytes. Default voice: George (`JBFqnCBsd6RMkjVDRZzb`), model: `eleven_multilingual_v2`.
- `router.py` — FastAPI router mounted at `/voice`:
  - `POST /voice/stt` — multipart `audio` upload → `{"text": "..."}` 
  - `POST /voice/tts` — `{"text": "..."}` → `audio/mpeg` response

**Frontend (`frontend/`):**
- Next.js 15 App Router scaffold with Tailwind CSS dark theme
- `components/AudioRecorder.tsx` — MediaRecorder API → webm blob → `/api/voice/stt`
- `components/ChatUI.tsx` — text + voice input → `/api/agent/chat` → TTS playback
- `components/SourcePlayer.tsx` — renders chunk citations with episode ID + timestamp range
- API proxy in `next.config.ts` rewrites `/api/*` → `localhost:8000/*`

## Test counts
- STT: 3 tests (`test_voice_stt.py`)
- TTS: 4 tests (`test_voice_tts.py`)
- Router: 5 tests (`test_voice_router.py`)
- **Total backend: 89 passing**

## Key env vars
- `OPENAI_API_KEY` — required for STT (Whisper API)
- `ELEVENLABS_API_KEY` — required for TTS; 400 returned if missing

## Run frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev  # http://localhost:3000
```
