# Deploying to production (Vercel + Railway)

Split-domain deploy: frontend on Vercel, backend + Postgres/pgvector on Railway. The
backend is already CORS/proxy-ready for this split (see `DECISIONS.md`, Etap 13) — no
code changes needed, only account setup and environment variables.

## 1. Push to GitHub

Make sure `main` is pushed and the repo is **public** before sharing the link (Settings →
General → Danger Zone → Change visibility, on github.com).

## 2. Railway — Postgres with pgvector

Railway's default "PostgreSQL" template does **not** have the `vector` extension compiled
in — `CREATE EXTENSION vector` (run automatically by `init_db` on backend startup) will
fail against it. Instead:

1. New Project → **Deploy a Docker Image**.
2. Image: `pgvector/pgvector:pg16` (same image `docker-compose.yml` already uses locally).
3. Set a volume mount for `/var/lib/postgresql/data` so data survives restarts.
4. Set env vars on that service: `POSTGRES_USER=voicerag`, `POSTGRES_PASSWORD=<generate a strong one>`, `POSTGRES_DB=voicerag`.
5. Once running, note the private connection string Railway shows (`postgresql://voicerag:<password>@<service>.railway.internal:5432/voicerag`).

## 3. Railway — backend service

1. In the same project: **New Service → GitHub Repo** → select this repo.
2. Service settings → **Root Directory**: `backend`. Railway will pick up `backend/railway.toml` and `backend/Dockerfile` automatically.
3. Environment variables (Railway service → Variables):

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | the Postgres service's internal connection string from step 2 |
   | `OPENAI_API_KEY` | your key |
   | `ANTHROPIC_API_KEY` | your key |
   | `ELEVENLABS_API_KEY` | your key |
   | `COHERE_API_KEY` | your key (optional — omit to silently fall back to `hybrid`) |
   | `ALLOWED_ORIGINS` | placeholder for now, e.g. `https://placeholder.vercel.app` — fix in step 5 |
   | `RATE_LIMIT_ENABLED` | `true` |
   | `API_KEY` | optional — set only if you want to gate `/ingest/*` behind a shared secret |

4. Deploy. Railway assigns a public URL like `https://<service>.up.railway.app` — copy it.
5. Verify: `curl https://<service>.up.railway.app/health` → `{"status": "ok"}`.

## 4. Vercel — frontend

1. Import the GitHub repo as a new Vercel project.
2. Project Settings → **Root Directory**: `frontend`.
3. Environment variables (Project Settings → Environment Variables):

   | Variable | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | the Railway backend URL from step 3.4, e.g. `https://<service>.up.railway.app` |
   | `API_KEY` | same value as the backend's `API_KEY`, only if you set one — `middleware.ts` injects it server-side into `/api/ingest/*` requests |

4. Deploy. Vercel assigns a URL like `https://<project>.vercel.app`.

## 5. Close the loop: CORS

Back in Railway, update the backend's `ALLOWED_ORIGINS` to the real Vercel URL from step 4
(comma-separated if you also want to allow a custom domain later), then redeploy the
backend service. Without this, the browser blocks every request with a CORS error.

## 6. Smoke test

- Open the Vercel URL, ask a text question — should get a streamed answer with source citations.
- Try the mic button — should transcribe, answer, and speak the answer back.
- Switch the search-mode selector and confirm results change.
- `curl -X POST https://<backend>/ingest/rss -H "Content-Type: application/json" -d '{"url": "...", "limit": 1}'` (add `-H "X-API-Key: ..."` if you set one) to confirm ingest still works from the public backend.

## 7. Custom domain (optional)

Both Vercel and Railway support attaching a custom domain in their project settings — not
required, the default `*.vercel.app`/`*.up.railway.app` URLs are fine for a portfolio link.

## Cost

Pay-as-you-go APIs only (no fixed monthly fee for OpenAI/Anthropic/Cohere). Railway's free
trial credit or lowest paid tier (~$5/mo) covers a low-traffic Postgres + backend pair;
Vercel's free Hobby tier is enough for the frontend.

Rough API balance to keep funded (verify current rates on each provider's pricing page —
these move):

| Provider | Recommended balance | Why |
|---|---|---|
| OpenAI | $10 | Whisper transcription (~$0.006/min) for ingest + live voice-question STT |
| Anthropic | $15 | Agent chat (per visitor conversation) + eval LLM-as-judge sweep |
| ElevenLabs | $0 (free tier) | ~10k chars/mo free covers a low-traffic demo; upgrade (~$5/mo) only if traffic grows |
| Cohere | $0 (free trial key) | `hybrid+rerank` mode only, rate-limited trial key is enough for a demo |
