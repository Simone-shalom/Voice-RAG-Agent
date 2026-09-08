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
3. Add a **Volume**, but mount it at `/var/lib/postgresql/data/pgdata` (a subdirectory),
   **not** `/var/lib/postgresql/data` directly — and set `PGDATA=/var/lib/postgresql/data/pgdata`
   as an env var on the service. Railway's fresh volumes come with a pre-existing
   `lost+found` directory; `initdb` refuses to initialize into a non-empty directory, so
   mounting straight at `/var/lib/postgresql/data` crashes the container on first boot
   with `initdb: error: directory ".../data" exists but is not empty`. Hit this live —
   costs a few minutes to notice since the container looks "Active" in Railway's UI while
   actually crash-looping.
4. Set env vars on that service: `POSTGRES_USER=voicerag`, `POSTGRES_PASSWORD=<generate a strong one>`, `POSTGRES_DB=voicerag`.
5. Once running, note the private connection string Railway shows (`postgresql://voicerag:<password>@<service>.railway.internal:5432/voicerag`).

## 3. Railway — backend service

1. In the same project: **New Service → GitHub Repo**, paste the repo URL directly if it
   doesn't show up in the picker (a fresh Railway account often has no repos listed until
   its GitHub App is authorized — see step 2).
2. If the deployed service's Settings → Source shows **"GitHub Repo not found"** even
   though the repo is selected: Railway's GitHub App isn't installed on that repo yet.
   Go to **github.com/settings/installations** → find Railway → **Configure** → add the
   repo (or grant all-repos access) → save. Back in Railway, disconnect and reconnect the
   source repo (Settings → Source → the pencil icon) to force it to re-check access —
   editing Root Directory alone doesn't retrigger the check.
3. Service settings → **Root Directory**: `backend`. Railway will pick up `backend/railway.toml` and `backend/Dockerfile` automatically.
4. Environment variables (Railway service → Variables):

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | the Postgres service's internal connection string from step 2 |
   | `OPENAI_API_KEY` | your key |
   | `ANTHROPIC_API_KEY` | your key |
   | `ELEVENLABS_API_KEY` | your key (optional — voice answers just skip the spoken reply without it) |
   | `COHERE_API_KEY` | your key (optional — omit to silently fall back to `hybrid`) |
   | `ALLOWED_ORIGINS` | placeholder for now, e.g. `https://placeholder.vercel.app` — fix in step 5 |
   | `RATE_LIMIT_ENABLED` | `true` |
   | `PORT` | `8000` — **set this explicitly.** Generating a domain (next step) makes you pick a "target port," but Railway can independently assign a *different* random port as the container's actual `$PORT` at runtime — we hit exactly this (target port 8000, container actually listening on 8080) and every request 502'd with "Application failed to respond" even though the app itself had started fine. Setting `PORT` yourself pins both to the same value. |
   | `API_KEY` | optional — set only if you want to gate `/ingest/*` behind a shared secret |

5. Deploy, then Settings → Networking → **Generate Domain**, port `8000` (matching the `PORT` var above). Railway assigns a public URL like `https://<service>.up.railway.app` — copy it.
6. Verify: `curl https://<service>.up.railway.app/health` → `{"status": "ok"}`. If you get a 502 "Application failed to respond" but the deployment shows Active, check Deploy Logs for the actual `uvicorn running on http://0.0.0.0:<port>` line and make sure it matches the domain's target port (see the `PORT` row above).

## 4. Vercel — frontend

1. Import the GitHub repo as a new Vercel project (vercel.com/new → paste the repo URL
   directly if it's not in the picker list).
2. Project Settings → **Root Directory**: `frontend` — Vercel auto-detects it as Next.js.
3. Vercel auto-detects env vars from the repo's `.env.example`, which lists the
   **backend's** variables (`DATABASE_URL`, `OPENAI_API_KEY`, etc.) — none of those belong
   on the frontend project. Remove all of them and add only what's below.
4. Environment variables (Project Settings → Environment Variables):

   | Variable | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | the Railway backend URL from step 3.4, e.g. `https://<service>.up.railway.app` |
   | `API_KEY` | same value as the backend's `API_KEY`, only if you set one — `middleware.ts` injects it server-side into `/api/ingest/*` requests |

5. Deploy. Vercel assigns a URL like `https://<project>.vercel.app`.

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
