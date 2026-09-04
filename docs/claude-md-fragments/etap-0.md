## Etap 0 - Bootstrap

**Uruchomienie stacku:**
```
docker compose up
curl localhost:8000/health   # -> {"status": "ok"}
```

**Struktura repo:**
- `backend/` - FastAPI (Python 3.12)
- `frontend/` - Next.js (od Etapu 4)
- `mcp-server/` - MCP server (od Etapu 6)
- `eval/` - eval harness (od Etapu 5)
- `docs/superpowers/plans/` - plany per etap (writing-plans)
- `docs/claude-md-fragments/` - materialy do finalnego CLAUDE.md, scalane w Etapie 7
- `DECISIONS.md` - log decyzji technicznych z uzasadnieniami

**Konwencja:** kazdy etap konczy sie wpisem w `DECISIONS.md` i nowym plikiem `docs/claude-md-fragments/etap-N.md`.
