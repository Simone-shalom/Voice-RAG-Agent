# Architecture Walkthrough — notatki do nauki

Ten dokument to materiał do powrotów i nauki — pełny obraz architektury Voice Knowledge Agent,
dwóch głównych przepływów danych, oraz konkretny ślad wykonania zapytania multi-hop przez agenta.
Uzupełnienie `DECISIONS.md` (który tłumaczy *dlaczego*) i `CLAUDE.md` (który jest referencją API/configu).

---

## 1. Widok z lotu ptaka

```
┌─────────────┐      HTTP       ┌──────────────┐
│  Next.js     │ ──────────────> │  FastAPI      │
│  frontend    │ <────────────── │  backend      │
│  :3000       │                 │  :8000        │
└─────────────┘                 └──────┬───────┘
                                        │ SQL
                                 ┌──────▼───────┐
                                 │ Postgres 16   │
                                 │ + pgvector    │
                                 │ :5433         │
                                 └──────┬───────┘
                                        │ httpx (REST)
                                 ┌──────▼───────┐
                                 │  MCP server   │
                                 │ (osobny venv) │
                                 └──────────────┘
```

Monorepo, 4 niezależnie uruchamialne komponenty. Backend (`backend/app/main.py`) to jedna
aplikacja FastAPI z pięcioma routerami, każdy odpowiadający jednemu etapowi budowy:

```python
app.include_router(ingest_router)    # Etap 1 — upload/transkrypcja/chunking
app.include_router(search_router)    # Etap 2 — semantic/bm25/hybrid/rerank
app.include_router(agent_router)     # Etap 3 — LangGraph agent
app.include_router(voice_router)     # Etap 4 — STT/TTS
app.include_router(episodes_router)  # Etap 6 — lista/detale epizodów
```

Baza inicjalizowana jest w `lifespan`, nie na poziomie importu modułu:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield
```

**Dlaczego:** import-time `init_db` łączyłby się z Postgresem zanim kontener bazy skończy start
(race condition przy `docker compose up`) — worker padłby przed zbudowaniem `app`. `lifespan`
uruchamia się po starcie serwera, więc problem znika. Bonus: testy nie muszą patchować `init_db`
przed importem `app.main`.

**Dlaczego monorepo, a nie osobne repo per komponent** (Etap 0): jeden cykl życia, jeden deploy
target na etap. Osobne repo dodałoby koszt synchronizacji bez korzyści (brak wielu niezależnych
zespołów/wdrożeń).

---

## 2. Przepływ #1 — ingest (audio → przeszukiwalna wiedza)

```
POST /ingest/upload (plik) lub /ingest/url (link)
        │
        ▼
transcribe()          — Whisper API, zwraca segmenty {text, start, end}
        │
        ▼
chunk_segments()       — dzieli segmenty na chunki wg pauz mowy (>=0.5s) lub 300 słów
        │
        ▼
embed_texts()          — text-embedding-3-small, JEDNO batch-wywołanie na wszystkie chunki
        │
        ▼
Episode + Chunk[] → db.commit()
```

`backend/app/ingest/service.py`:

```python
def ingest_audio(audio_path, filename, source_url, db):
    segments = transcribe(audio_path)
    chunk_dicts = chunk_segments(segments)
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(texts)          # batch, nie pętla z N wywołaniami
    episode = Episode(...)
    db.add(episode)
    for chunk_dict, embedding in zip(chunk_dicts, embeddings):
        db.add(Chunk(episode_id=episode.id, start_ts=..., end_ts=..., text=..., embedding=embedding))
    db.commit()
```

`ingest_from_url` ściąga plik **strumieniowo** do pliku tymczasowego (`httpx` `.stream("GET", url)`
+ `iter_bytes()`) zamiast ładować cały plik do pamięci, i sprząta go w `finally` niezależnie od wyniku.

### Chunker — `backend/app/ingest/chunker.py`

```python
PAUSE_THRESHOLD = 0.5   # sekundy — pauza dłuższa niż to = granica chunka
MAX_CHUNK_WORDS = 300    # twardy limit na wypadek długiej wypowiedzi bez pauzy
```

Dzieli segmenty Whispera na chunki przy naturalnych pauzach mowy, nie po stałej liczbie
znaków/tokenów (jak `CharacterTextSplitter`). Pusty segment (whitespace-only, marker ciszy z
Whispera) jest odrzucany przed embeddingiem — pusty string wysłany do OpenAI Embeddings API
zwraca HTTP 400.

**Dlaczego pauzy, nie stały rozmiar:** granica chunka przy pauzie daje semantycznie spójne
fragmenty i zachowuje precyzję `start_ts`/`end_ts` do cytowania. Fixed-size cięłoby zdania w pół.

### Invariant nr 1 systemu

`start_ts`/`end_ts` powstają **wyłącznie tutaj**, w `chunk_segments`. Każda kolejna warstwa —
search, agent, frontend `SourcePlayer` — tylko je czyta i przekazuje dalej. Złam to raz tutaj, a
cała funkcja "cytatów z timestampem" w systemie się psuje.

---

## 3. Model danych

`backend/app/db/models.py`:

```python
class Episode(Base):
    id, filename, source_url (nullable), created_at
    chunks = relationship("Chunk", back_populates="episode")

class Chunk(Base):
    id, episode_id (FK), start_ts: Float, end_ts: Float, text: Text,
    embedding: Vector(1536) | None
```

- `embedding` jest **nullable** — chunk może istnieć i być znaleziony przez BM25 zanim/nawet gdyby
  embedding się nie wygenerował. Oddziela porażkę jednej ścieżki (embeddingi) od integralności rekordu.
  Wymusza to guard `WHERE embedding IS NOT NULL` w każdym zapytaniu semantycznym.
- `tsvector` do BM25 żyje jako **generated stored column** bezpośrednio w `chunks`
  (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`, indeks GIN) — Postgres utrzymuje
  go zawsze aktualnym automatycznie, bez triggera ani batch-reindexu.

---

## 4. Przepływ #2 — zapytanie użytkownika (od pytania do odpowiedzi)

```
[opcjonalnie: mikrofon] → POST /voice/stt → tekst pytania
        │
        ▼
POST /agent/chat {message, thread_id}
        │
        ▼
build_graph(db, llm).invoke({messages: [pytanie], step_count: 0})
        │
   ┌────▼────┐
   │  agent   │◄──────────────┐
   │  (LLM)   │                │
   └────┬────┘                │
        │ tool_calls?          │
      tak│                     │
        ▼                     │
   ┌─────────┐                │
   │  tools   │────────────────┘
   └─────────┘
        │ nie (LLM ma dość informacji)
        ▼
   odpowiedź tekstowa + `steps`
        │
        ▼ [opcjonalnie]
POST /voice/tts → mp3 (ElevenLabs)
```

### Hybrid search — `backend/app/search/searcher.py`

```python
def hybrid_search(query, limit, db):
    fetch = limit * 2                                    # headroom dla fuzji
    sem = semantic_search(query, limit=fetch, db=db)      # pgvector cosine, "<=>"
    bm25 = bm25_search(query, limit=fetch, db=db)          # tsvector/ts_rank
    fused = reciprocal_rank_fusion([sem, bm25])            # łączy RANGI, nie score'y
    return [...][:limit]
```

`fetch = limit * 2` — każdy sub-searcher zwraca 2× więcej niż finalnie potrzeba, żeby RRF miało z
czego fuzjonować (gdyby oba zwracały dokładnie `limit` i się nie pokrywały, straciłbyś połowę sygnału).

`semantic_search` (SQL, `<=>` = cosine distance pgvector):

```sql
SELECT ..., 1 - (c.embedding <=> :emb::vector) AS similarity
FROM chunks c
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> :emb::vector
LIMIT :limit
```

`bm25_search` (SQL, `plainto_tsquery` — toleruje naturalne pytania, nie wymaga operatorów):

```sql
SELECT ..., ts_rank(c.text_search, query) AS bm25_score
FROM chunks c, plainto_tsquery('english', :query_text) query
WHERE c.text_search @@ query
```

`reciprocal_rank_fusion` (`search/rrf.py`):

```python
scores[cid] += 1.0 / (k + rank)    # k=60, suma odwrotności RANGI, nie score'u
```

**Dlaczego rangi, nie surowe score'y:** cosine similarity i `ts_rank` są na niekompatybilnych
skalach — trzeba by kalibrować wagę alfa między nimi. RRF omija to całkowicie: "chunk był #1 w
semantic i #3 w BM25" łączy się bez znajomości co te liczby oryginalnie znaczyły. `k=60` to
standardowa wartość z Cormack et al. 2009 — nic do tuningu.

`hybrid+rerank` dokłada Cohere Rerank jako opcjonalny finalny przebieg — wymaga
`COHERE_API_KEY`, brak klucza = czytelny HTTP 400 (nie cichy fallback do zwykłego hybrid).

### LangGraph — `backend/app/agent/graph.py`

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # akumuluje, nie nadpisuje
    step_count: int

def agent_node(state, config):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}

def should_continue(state):
    if state["step_count"] >= MAX_STEPS: return END        # MAX_STEPS = 6, guard przed pętlą
    if last.tool_calls: return "tools"
    return END

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")   # po narzędziu ZAWSZE wraca do agenta
```

`Annotated[list[BaseMessage], add_messages]` to reducer LangGraph — nowe wiadomości są dopisywane
do listy, nie ją zastępują. Dzięki temu cała historia (pytanie → wywołanie narzędzia → wynik →
kolejne wywołanie → finalna odpowiedź) jest widoczna w każdej kolejnej turze `agent_node`.

`ToolNode(tools)` (z `langgraph.prebuilt`) sam odczytuje `tool_calls` z ostatniej wiadomości,
woła odpowiednią funkcję Pythona (closure nad `db`) i wstrzykuje wynik jako `ToolMessage`.

`MemorySaver` + `thread_id` (`agent/router.py`) daje pamięć konwersacji między turami bez
przechowywania historii po stronie frontendu.

**Dlaczego graf z conditional edges, nie sztywny chain:** LLM decyduje w każdej iteracji czy
wywołać narzędzie czy zakończyć. Ta sama architektura obsługuje pytanie proste (1 krok) i
multi-hop (2+ kroki) bez hardkodowania ścieżki po typie pytania. DoD test Etapu 3 weryfikuje to
explicité przez porównanie `step_count` między dwoma typami pytań.

### Cztery narzędzia agenta — `backend/app/agent/tools.py`

| Narzędzie | Co robi | Mechanizm |
|---|---|---|
| `search_transcripts(query, limit=5)` | hybrid search po wszystkich transkrypcjach | woła `hybrid_search()` |
| `get_context_around_timestamp(episode_id, timestamp, window_seconds=30)` | rozszerza kontekst wokół już znalezionego cytatu | zwykłe SQL po `start_ts`/`end_ts` w oknie, sortowane chronologicznie — **nie** hybrid search, bo tu liczy się bliskość czasowa, nie trafność |
| `compare_across_episodes(query, episode_ids, limit_per_episode=3)` | wyszukiwanie pogrupowane per epizod | `hybrid_search` + grupowanie wyników po `episode_id` |
| `summarise_segment(text)` | formatuje surowy fragment do dalszej analizy przez LLM | obcina do `MAX_CONTEXT_CHARS=2000`, brak wywołania zewnętrznego API |

Wyniki nigdy nie zawierają pełnych transkrypcji — zawsze `{chunk + timestamp + episode}`
(`_format_chunks`), co utrzymuje kontekst LLM krótki i przewidywalny co do kosztu.

---

## 5. Ślad wykonania — konkretne zapytanie multi-hop

Pytanie: **„Co podcast mówi o pruningu sieci neuronowych, i czy możesz dać mi więcej kontekstu
wokół pierwszej wzmianki?"** — dwie części, wymaga wyszukiwania + doprecyzowania → naturalnie 3
przejścia przez pętlę grafu.

**Stan początkowy:** `{"messages": [HumanMessage("Co podcast mówi o pruningu...")], "step_count": 0}`

**Krok 1 — `agent_node`:** LLM dostaje samo pytanie, nic jeszcze nie wie o treści podcastów.
Zwraca wiadomość z `tool_calls=[{"name": "search_transcripts", "args": {"query": "pruning sieci
neuronowych", "limit": 5}}]`. `step_count → 1`. `should_continue` widzi `tool_calls` → `tools`.

**`tools`:** `search_transcripts(...)` → `hybrid_search()` → `semantic_search` (10 kandydatów) +
`bm25_search` (10 kandydatów) → RRF → top 5, sformatowane:
```
[ep-a1b2 @ 412.3s–438.7s] ...mówimy tu o pruningu jako sposobie na redukcję wag...
[ep-a1b2 @ 890.1s–905.4s] ...po pruningu model stracił tylko 2% accuracy...
```
Dopisane jako `ToolMessage` do `messages` (reducer `add_messages` — nic nie jest nadpisywane).

**Krok 2 — `agent_node`:** LLM widzi pytanie + wynik wyszukiwania, zna `episode_id=ep-a1b2` i
pierwszy timestamp `412.3s`. Druga część pytania wymaga innego narzędzia — woła
`get_context_around_timestamp(episode_id="ep-a1b2", timestamp=412.3, window_seconds=30.0)`.
`step_count → 2`.

**`tools`:** to **nie** jest hybrid search — zwykłe SQL po `start_ts`/`end_ts` w oknie ±15s wokół
`412.3`, sortowane chronologicznie. Zwraca 3-4 sąsiadujące chunki jako jeden blok tekstu.

**Krok 3 — `agent_node`:** LLM ma komplet (oryginalne wyszukiwanie + rozszerzony kontekst).
Generuje finalną odpowiedź tekstową, bez `tool_calls`. `step_count → 3`. `should_continue` → `END`.

**Odpowiedź:** `{"reply": "...", "steps": 3}`.

Dla porównania: „Cześć, co potrafisz?" nie wywołuje żadnego narzędzia → `steps: 1` (jeden
przebieg `agent_node`, od razu `END`). To dokładnie ta różnica, którą DoD test Etapu 3 sprawdza —
porównanie `step_count` między pytaniem prostym a multi-hop.

---

## 6. Frontend

`frontend/app/page.tsx`: sidebar (`FileUploader`) + główny panel (`ChatUI`).

| Komponent | Rola |
|---|---|
| `FileUploader` | upload plików/URL → `POST /ingest/*`, lista epizodów → `GET /episodes` |
| `ChatUI` | wysyła pytania → `POST /agent/chat`, renderuje odpowiedzi z cytatami |
| `AudioRecorder` | nagrywa mikrofon → `POST /voice/stt` → tekst trafia do `ChatUI` |
| `SourcePlayer` | odtwarza fragment audio od konkretnego `start_ts` — konsument invariantu z sekcji 2 |

Next.js App Router, bez własnej logiki serwerowej — czysty klient uderzający w FastAPI po
`NEXT_PUBLIC_API_URL`. Cała logika biznesowa (agent, search) i tak żyje w Pythonie, więc route
handlers w Next.js dodałyby tylko warstwę proxy bez wartości.

---

## 7. MCP server — osobny proces, osobny venv

`mcp-server/tools.py` woła backend przez `httpx`, **nie** importuje `backend/app` bezpośrednio:

```python
def search_knowledge_base(query, limit=5) -> str:
    r = httpx.get(f"{BASE_URL}/search", params={"q": query, "mode": "hybrid", "limit": limit})
```

**Dlaczego:** twardy konflikt zależności — `mcp>=1.0` wymaga `starlette>=1.0`, `fastapi==0.115.0`
wymaga `starlette<0.39`. Nierozwiązywalne w jednym venv bez ryzykownego upgrade'u FastAPI.
Rozdzielenie na dwa venv jest naturalne — MCP server i tak jest osobnym procesem w runtime.

Efekt uboczny na korzyść: MCP zawsze dostaje te same wyniki (ta sama fuzja RRF, ten sam
reranking) co frontend, bo oba biją w ten sam endpoint — zero ryzyka rozjazdu logiki między
dwiema kopiami kodu wyszukiwania.

Trzy narzędzia MCP (`mcp-server/tools.py`):

| Narzędzie | Zwraca |
|---|---|
| `search_knowledge_base(query, limit=5)` | sformatowana lista chunków z timestampami |
| `get_episode_summary(episode_id)` | metadane epizodu + pierwsze 10 chunków transkrypcji |
| `find_mentions(topic, limit=10)` | lista chunków gdzie temat jest wspomniany |

---

## 8. Eval harness — `eval/`

RAG triad: **Context Relevance**, **Groundedness**, **Answer Relevance**, każda 0.0-1.0, oceniana
przez LLM-as-judge (Claude Haiku, `eval/scorer.py::make_anthropic_judge`).

Golden dataset (`eval/golden_dataset/questions.json`) przechowuje `expected_topics: [str]`, nie
gotowe odpowiedzi referencyjne — bo nie ma prawdziwego podcastu do wygenerowania "ground truth"
bez circularity. Sędzia dostaje prompt proszący o liczbę całkowitą 1-5 (małe modele są bardziej
deterministyczne na pojedynczej cyfrze niż na foacie), `_parse_score` wyciąga pierwszą cyfrę ze
stringa (odporność na "The score is 4." zamiast czystego "4") i normalizuje `(val-1)/4.0`.

```bash
python -m eval.evaluate \
  --base-url http://localhost:8000 \
  --modes semantic bm25 hybrid hybrid+rerank \
  --output docs/eval_report
```

15 pytań × 4 tryby wyszukiwania × 3 metryki = 180 wywołań sędziego na pełny przebieg.

---

## 9. Testowanie i wdrożenie

- **131 testów, wszystkie mockowane** — żaden nie wymaga żywej bazy ani prawdziwych kluczy API.
  Klienci zewnętrzni (OpenAI, Cohere, Anthropic, httpx do ElevenLabs) są `lru_cache`-owane
  singletony, więc `patch("...module._client", return_value=mock)` podmienia je w jednym miejscu.
- **`docker-compose up`** stawia 3 kontenery: `postgres` (`pgvector/pgvector:pg16`, healthcheck
  `pg_isready`, backend czeka na `condition: service_healthy`), `backend` (hot-reload przez
  `./backend:/app`), `frontend` (analogicznie, osobny wolumen na `node_modules`).
- **Konwencja gita:** granularne commity per-task na branchu roboczym (`etap-N-<nazwa>`), potem
  jeden squash-commit na `main` per cały etap (`git reset --soft $(git merge-base HEAD main)`).
  `git log` na `main` czyta się jako historia etapów projektu; granularna historia żyje na
  branchu do czasu jego usunięcia.

---

## 10. Modele użyte w projekcie — szybka tabela referencyjna

| Warstwa | Model | Dlaczego ten |
|---|---|---|
| STT (ingest + voice) | OpenAI `whisper-1` (API) | word/segment timestamps "z pudełka", zero zarządzania GPU na dev-machine |
| Embeddingi | `text-embedding-3-small` (1536-dim) | 5× tańszy od `-large`, wystarczający dla transkrypcji mówionego języka |
| Leksykalne wyszukiwanie | Postgres `tsvector`/`ts_rank` | już w stacku (obok pgvector), zero nowego serwisu, generated column = zawsze aktualne |
| Reranking (opcjonalny) | Cohere `rerank-english-v3.0` | cross-encoder bez potrzeby GPU, drop-in interfejs (query+docs→scores) |
| Agent LLM | Claude Haiku `claude-haiku-4-5-20251001` | najszybszy model z pełnym tool use — kluczowe bo pętla agenta mnoży wywołania LLM |
| LLM-as-judge (eval) | Claude Haiku (ten sam model) | ocena 1-5 to prostsze zadanie niż generowanie — Haiku wystarcza przy 180 wywołaniach/run |
| TTS | ElevenLabs `eleven_multilingual_v2` (REST, nie SDK) | SDK Pythona na Windows psuje się przez limit długości ścieżki (260 znaków) |

**Wzorzec widoczny we wszystkich wyborach:** żaden nie wymaga GPU na dev-machine, żaden nie
dokłada nowego serwisu jeśli coś już jest w stacku, i każdy ma na tyle odizolowany interfejs
(jeden plik, jasna sygnatura wejście/wyjście), że wymiana na innego dostawcę później to zmiana
lokalna, nie refaktor.

Pełne uzasadnienia z odrzuconymi alternatywami → `DECISIONS.md`.
