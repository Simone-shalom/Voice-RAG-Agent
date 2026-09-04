# Decisions Log

Log kazdej nietrywialnej decyzji technicznej podjetej w projekcie: co wybrano, jakie byly odrzucone alternatywy i dlaczego. Wpis powstaje w kroku 7 kazdego etapu (patrz `docs/superpowers/plans/2026-09-04-voice-rag-agent-roadmap.md`), zanim etap zostanie uznany za zamkniety.

To jest material zrodlowy pod pytania rekrutacyjne typu "dlaczego wybrales X, a nie Y" - wpis ma miec sens dla kogos, kto nie siedzial przy tej decyzji.

## Format wpisu

```
## [Etap N] Krotki tytul decyzji (YYYY-MM-DD)

**Decyzja:** co dokladnie wybrano.

**Alternatywy odrzucone:** co jeszcze bylo brane pod uwage i dlaczego odpadlo.

**Uzasadnienie:** dlaczego wybrana opcja wygrala - kompromis, ograniczenie, pomiar.
```

---

## [Etap 0] Struktura repo i nazewnictwo katalogow (2026-09-04)

**Decyzja:** monorepo z katalogami `backend/` (FastAPI), `frontend/` (Next.js), `mcp-server/`, `eval/`, `docs/` w jednym repozytorium.

**Alternatywy odrzucone:** osobne repozytoria per komponent (backend/frontend/mcp-server).

**Uzasadnienie:** projekt portfolio o jednym cyklu zycia, jeden deploy target na etap; osobne repo dodaloby koszt synchronizacji bez korzysci (brak wielu niezaleznych zespolow/wdrozen).
