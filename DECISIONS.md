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

---

## [Etap 1] OpenAI Whisper API vs. lokalny faster-whisper (2026-09-05)

**Decyzja:** OpenAI Whisper API (`whisper-1`, `response_format="verbose_json"`, `timestamp_granularities=["word","segment"]`).

**Alternatywy odrzucone:** `faster-whisper` (CTranslate2, lokalny, bez GPU na dev-machine).

**Uzasadnienie:** API daje `word_timestamps` out-of-the-box bez konfiguracji CUDA/cuDNN, eliminuje zarzadzanie modelem na dev-machine i jest wystarczajace dla 20-30 min nagrania. Lokalny model mozna podpiac pozniej przez podmiane `transcriber.py` — interfejs (`list[dict]` z `text/start/end`) pozostaje ten sam.

---

## [Etap 1] Chunking wg pauz mowy vs. liczba znakow/tokenow (2026-09-05)

**Decyzja:** dzielenie chunkow na podstawie przerw miedzy segmentami Whispera (`PAUSE_THRESHOLD = 0.5s`) z gornym limitem slow (`MAX_CHUNK_WORDS = 300`). Segmenty tylko ze spacjami/bialymi znakami sa odrzucane przed embeddingiem.

**Alternatywy odrzucone:** chunking po stalej liczbie znakow (langchain `CharacterTextSplitter`) lub tokenow.

**Uzasadnienie:** Whisper segmentuje mowe naturalnie przy pauzach — granica chunka przy pauzie daje semantycznie spojne fragmenty i zachowuje `start_ts`/`end_ts` potrzebne do cytowania timestampow. Podejscie "fixed-size" cialby zdania w pol i psulo precyzje cytatow. Odrzucanie pustych segmentow zabezpiecza przed HTTP 400 z OpenAI Embeddings (puste stringi sa odrzucane przez API).

---

## [Etap 1] pgvector cosine similarity (`<=>`) vs. inne metryki (2026-09-05)

**Decyzja:** cosine similarity przez operator `<=>` pgvector; `WHERE c.embedding IS NOT NULL` w zapytaniu; `text-embedding-3-small` (1536 dim).

**Alternatywy odrzucone:** L2 distance (`<->`), inner product (`<#>`); `text-embedding-3-large` (3072 dim).

**Uzasadnienie:** cosine similarity jest standardem dla text embeddings (nie zalezi od normy wektora). Guard `IS NOT NULL` zapobiega cichemu pomijaniu chunkow bez embeddingu. `text-embedding-3-small` jest 5x tanszy niz large i wystarczajacy dla semantic search na transkrypcjach audio — mozna podmienic pozniej przez zmiane stalej `EMBEDDING_MODEL` i migracje wymiarow kolumny.

---

## [Etap 1] FastAPI lifespan vs. module-level init_db (2026-09-05)

**Decyzja:** `init_db(engine)` wywolywany w `@asynccontextmanager async def lifespan(app)`, nie na poziomie importu modulu.

**Alternatywy odrzucone:** `init_db(engine)` bezposrednio na poziomie modulu `app/main.py`.

**Uzasadnienie:** wywolanie na poziomie importu lacze sie z baza w momencie startu workera — jesli Postgres nie skonczyl inicjalizacji (race condition), worker pada przed zbudowaniem `app`. Lifespan uruchamia sie po starcie serwera, co eliminuje problem. Dodatkowy bonus: testy nie potrzebuja juz `with patch("app.db.session.init_db"):` przed importem `app.main`.

---

## [Etap 3] LangGraph StateGraph vs. chain (2026-09-05)

**Decyzja:** LangGraph `StateGraph` z `AgentState` (messages + step_count), dwoma węzłami (agent, tools) i conditional edges. `MAX_STEPS = 6` jako guard przeciwko pętli. `MemorySaver` do checkpointingu historii rozmowy per `thread_id`.

**Alternatywy odrzucone:** LangChain `AgentExecutor` (deprecated); hardkodowany chain z ustaloną sekwencją kroków; ReAct pattern bez LangGraph.

**Uzasadnienie:** Conditional edges w LangGraph pozwalają agentowi decydować w każdej iteracji: wywołać narzędzie czy zakończyć. To nie jest chain — ta sama architektura obsługuje 1 krok (proste pytanie) i 2+ kroki (multi-hop) bez hardkodowania ścieżki. DoD test weryfikuje to eksplicite przez porównanie `step_count`. `MAX_STEPS` zapobiega nieskończonej pętli bez konieczności śledzenia zewnętrznego stanu.

---

## [Etap 3] Cohere vs. Anthropic (claude-haiku) jako LLM agenta (2026-09-05)

**Decyzja:** Claude claude-haiku-4-5-20251001 (przez `langchain-anthropic`) jako LLM w endpoincie `/agent/chat`. Klucz API (`ANTHROPIC_API_KEY`) sprawdzany przy każdym request, nie w lifespan.

**Alternatywy odrzucone:** GPT-4o-mini (OpenAI); Cohere Command R; weryfikacja klucza przy starcie aplikacji.

**Uzasadnienie:** Haiku to najszybszy model Claude z pełną obsługą tool use — kluczowe dla agentowej latencji. Weryfikacja klucza przy request (nie starcie) pozwala aplikacji działać bez `ANTHROPIC_API_KEY` w środowiskach gdzie agent nie jest używany (np. testy, dev bez kluczy). HTTP 400 z czytelnym komunikatem zamiast crashu przy starcie.

---

## [Etap 2] BM25 via PostgreSQL tsvector vs. external Elasticsearch/Typesense (2026-09-05)

**Decyzja:** BM25 przez natywny `tsvector`/`ts_rank` PostgreSQL jako generated stored column (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`) z indeksem GIN.

**Alternatywy odrzucone:** osobny silnik Elasticsearch lub Typesense; biblioteka Python `rank_bm25` (in-memory); `websearch_to_tsquery` zamiast `plainto_tsquery`.

**Uzasadnienie:** PostgreSQL jest juz w stacku — dodanie `tsvector` nie wymaga nowego serwisu. Generated column sprawia, ze kolumna jest zawsze aktualna bez triggera czy backfillingu. `plainto_tsquery` lepiej toleruje naturalne zapytania (brak wymagania operatorow); GIN index daje O(log n) lookup. Kompromis: ts_rank nie jest identyczny z BM25 Okapi (brak normalizacji dlugosci dokumentu domyslnie), ale dla transkrypcji audio o podobnej dlugosci chunkow roznica jest pomijalnie mala.

---

## [Etap 2] Reciprocal Rank Fusion (RRF) vs. score normalization (2026-09-05)

**Decyzja:** RRF z k=60 jako metoda fuzji semantic + BM25. Kazda lista przekazuje rangi (1, 2, 3...), nie surowe score. Suma `1/(k+rank)` per lista.

**Alternatywy odrzucone:** normalizacja min-max score + srednia wazona; Borda Count; liniowa kombinacja score z waga alfa.

**Uzasadnienie:** RRF nie wymaga kalibracji wagi alfa miedzy cosine similarity a ts_rank (sa na roznych skalach i nie sa bezposrednio porownywalnych). k=60 to standardowa wartosc z pracy Cormack et al. 2009 — dobrze dziala w praktyce bez tuningu. Fetch 2x limit z kazdego sub-searcher daje wystarczajacy headroom dla fuzji bez nadmiernego kosztu zapytan.

---

## [Etap 2] Cohere Rerank vs. lokalny cross-encoder (2026-09-05)

**Decyzja:** Cohere Rerank API (`rerank-english-v3.0`) jako opcjonalna warstwa reranking. Wymagany `COHERE_API_KEY`; brak klucza zwraca HTTP 400 z czytelnym komunikatem (nie cichy fallback).

**Alternatywy odrzucone:** lokalny cross-encoder (HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2`); zawsze-wlaczony reranking w trybie hybrid.

**Uzasadnienie:** Cross-encoder wymaga GPU lub wolnego inference na CPU — dev-machine nie ma GPU. Cohere API jest drop-in replacement i latwo wymienialny pozniej (ten sam interfejs: query + documents → scores). Opcjonalnosc przez zmienna srodowiskowa zapewnia, ze hybrid bez reranking dziala w kazdy srodowisku bez kluczy zewnetrznych.

---

## [Etap 1] Konwencja: 1 squash-commit per etap na `main` (2026-09-05)

**Decyzja:** granularne commity (per-task) tworzone sa na branchu roboczym podczas developmentu i code-review; przed scaleniem z `main` caly branch jest sciskany do jednego commita z pelnym opisem zmian.

**Alternatywy odrzucone:** scalanie wszystkich granularnych commitow do `main`; squash-merge przez GitHub.

**Uzasadnienie:** `git log` na `main` ma odzwierciedlac strukture etapow projektu — jeden commit = jeden etap = jedna sposta zmiana. Granularne commity sa dostepne w historii brancha do czasu jego usuniecia, co wystarcza do code-review i debugowania podczas trwania etapu.
