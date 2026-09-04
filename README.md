# Voice Knowledge Agent

> Praca w toku. Ten README wypelnia sie etapami - finalna struktura (patrz `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md`, Etap 7) bedzie zawierac: jednozdaniowy opis, GIF demo, diagram architektury, sekcje Evaluation z wynikami, decyzje techniczne, MCP server + GIF, setup lokalny, znane ograniczenia.

## Status

- [x] Etap 0 - Bootstrap repo
- [ ] Etap 1 - Ingest pipeline + baza
- [ ] Etap 2 - Hybrid search + reranking
- [ ] Etap 3 - LangGraph Agent
- [ ] Etap 4 - Warstwa glosowa
- [ ] Etap 5 - Eval harness
- [ ] Etap 6 - MCP Server
- [ ] Etap 7 - Deploy, README, polish

## Setup lokalny (na razie)

```bash
docker compose up
curl localhost:8000/health
```

Plan pelny: [`docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md`](docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md)
Log decyzji: [`DECISIONS.md`](DECISIONS.md)
