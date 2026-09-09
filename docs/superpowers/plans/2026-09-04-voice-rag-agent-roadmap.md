# Voice Knowledge Agent — Roadmap realizacji (Claude Code, agent team)

> **Dla agentów wykonawczych:** to jest plan orkiestracyjny (roadmap), nie plan TDD jednej funkcji.
> Każdy **Etap** poniżej wykonuje się osobno, w chwili gdy do niego dochodzimy, przez:
> 1. **superpowers:writing-plans** — Planner tworzy szczegółowy plan TDD tego etapu (`docs/superpowers/plans/YYYY-MM-DD-etap-N-<nazwa>.md`) z dokładnymi plikami, kodem, komendami testowymi.
> 2. **superpowers:subagent-driven-development** — Implementer(zy): osobny świeży subagent na każdy task z planu etapu, TDD, commit po tasku.
> 3. **code-review** (skill) — recenzja całego etapu po zakończeniu tasków.
> 4. **superpowers:verification-before-completion** — dowód działania przed ogłoszeniem "gotowe" (testy + smoke test manualny).
> 5. **superpowers:finishing-a-development-branch** — merge/PR do main.
>
> Ten dokument NIE zawiera jeszcze kroku 1 dla etapów 1-7 (będą pisane etap po etapie, tuż przed wykonaniem, bo szczegóły zależą od decyzji podjętych w etapach wcześniejszych). Zawiera za to cel, zakres, definition of done i pułapki każdego etapu — czyli spec, z którego Planner odpali writing-plans.

**Cel projektu:** Agent głosowy nad biblioteką nagrań audio (podcasty/wykłady) — transkrypcja, indeksowanie, wyszukiwanie hybrydowe, odpowiedzi głosowe z cytowaniem konkretnego timestampu w źródle. Portfolio projekt pod specjalizację Voice AI Engineer.

**Architektura:** FastAPI (backend, Python) + Next.js (frontend) + PostgreSQL/pgvector (hybrid search) + Whisper (STT) + ElevenLabs (TTS) + LangGraph (agent z narzędziami) + własny MCP server + eval harness (RAG triad, LLM-as-judge). Diagram pełny: patrz `Kontekst i uzasadnienie` niżej (wklejony z briefu źródłowego, nie duplikuję tu ASCII-artu).

**Tech stack:** Python 3.12, FastAPI, LangGraph, SQLAlchemy/psycopg, PostgreSQL + pgvector, OpenAI Whisper (lub `faster-whisper`), ElevenLabs API, Next.js 15 (App Router), Vercel AI SDK, Docker Compose, MCP Python SDK, pytest.

---

## Kontekst i uzasadnienie (dlaczego ten projekt)

Trzy dotychczasowe projekty komercyjne dotykały audio AI (Whisper w PANS; Whisper+TTS w projekcie implantów; ElevenLabs+RAG w projekcie transportowym) — to nie przypadek, tylko zaczątek specjalizacji. Rynek "AI Engineer, który zrobił RAG na PDF-ach" jest zatłoczony; rynek "Voice AI Engineer, który rozumie latencję konwersacji, VAD, streaming STT, orkiestrację workerów pod media" jest wąski i płatny lepiej. Czwarty projekt w tym obszarze zmienia narrację CV z "fullstack, który dotykał AI" na "inżynier od systemów głosowych AI".

Kontrargument dla równowagi: specjalizacja zawęża pulę ofert "Voice AI Engineer" względem ogólnych "AI Engineer". Ale RAG, LangGraph, MCP z tego projektu liczą się też w ofertach ogólnych — więc zawężenie dotyczy tylko *narracji*, nie *kwalifikowalności*.

**Priorytet dźwigni rekrutacyjnej (gdy zabraknie czasu, tnij w tej kolejności):**
- Nie do ruszenia: ingest + hybrid search + LangGraph agent + **eval harness** + **MCP server**
- Można uprościć: warstwa głosowa (tekst + TTS bez STT z mikrofonu wystarczy jako fallback)
- Można pominąć: `compare_across_episodes`, dopracowany frontend, deploy produkcyjny (Docker Compose + wideo demo wystarczy)

Eval harness i MCP server są ważniejsze niż ładne demo — ładnych demo jest pełno, zmierzonej jakości retrievalu i własnego MCP servera prawie nikt nie ma w portfolio na poziomie mid.

---

## Standardowy przebieg etapu (agent team)

Powtarzany dla każdego etapu 1-7 (Etap 0 jest lżejszy, patrz niżej):

| Krok | Kto / co | Output |
|---|---|---|
| 0. Brainstorm (tylko jeśli spec etapu poniżej zostawia realną niejednoznaczność projektową) | `superpowers:brainstorming` | decyzja zapisana w `DECISIONS.md` |
| 1. Plan szczegółowy | Planner uruchamia `superpowers:writing-plans` na podstawie sekcji "Zakres" i "Definition of Done" tego etapu | `docs/superpowers/plans/YYYY-MM-DD-etap-N-<nazwa>.md` — pełny plan TDD z kodem |
| 2. Izolacja | `superpowers:using-git-worktrees` lub branch `etap-N-<nazwa>` | osobny workspace/branch |
| 3. Implementacja | `superpowers:subagent-driven-development` — świeży subagent-implementator na task, czerwony→zielony→refaktor, commit po tasku | zaimplementowany kod + testy zielone |
| 4. Code review | skill `code-review` (poziom `medium`, przy Etapach 5 i 6 `high` — to one-way door dla jakości/architektury MCP) | lista findings, zaaplikowane poprawki (`--fix` lub ręcznie) |
| 5. Weryfikacja | `superpowers:verification-before-completion` — pełny test suite + smoke test manualny wg "Definition of Done" | dowód działania (log komend + wynik) |
| 6. Domknięcie | `superpowers:finishing-a-development-branch` | merge do `main` |
| 7. Dokumentacja | ręcznie/przez implementera: wpis w `DECISIONS.md` (decyzja + odrzucone alternatywy + dlaczego) + nowy plik `docs/claude-md-fragments/etap-N.md` (co z tego etapu powinno wejść do finalnego `CLAUDE.md`: komendy, konwencje, gotchas) | 2 pliki zaktualizowane |

Bramka przed przejściem do kolejnego etapu: testy zielone, code review bez otwartych findingów blokujących, `DECISIONS.md` i fragment CLAUDE.md zaktualizowane, smoke test z Definition of Done odtworzony ręcznie i potwierdzony.

---

## Etap 0 — Bootstrap projektu

**Cel:** szkielet repo, na którym da się zacząć etap 1 bez przestojów na konfigurację.

**Zakres:**
- `git init`, repo na GitHub (prywatne na start, publiczne przed Etapem 7)
- Struktura katalogów: `backend/` (FastAPI), `frontend/` (Next.js), `mcp-server/`, `eval/`, `docs/`
- `docker-compose.yml` szkielet: serwis `postgres` (obraz z pgvector, np. `pgvector/pgvector:pg16`), serwis `backend` (placeholder Dockerfile, health endpoint `/health`)
- `DECISIONS.md` w root — nagłówek + format wpisu (data, decyzja, alternatywy odrzucone, uzasadnienie)
- `docs/claude-md-fragments/` — katalog na fragmenty per etap
- `README.md` szkielet z sekcjami-nagłówkami do wypełnienia w Etapie 7 (patrz struktura w Etapie 7)
- `.gitignore` (Python, Node, `.env`, `*.mp3/*.wav` w katalogu ingest jeśli lokalny)
- `.env.example` z placeholderami na `OPENAI_API_KEY`/klucz Whisper, `ELEVENLABS_API_KEY`, `DATABASE_URL`

**Definition of Done:** `docker compose up` startuje `postgres` + `backend` (placeholder) bez błędów; `curl localhost:8000/health` zwraca 200; pierwszy commit na `main`; `DECISIONS.md` i `docs/claude-md-fragments/` istnieją i są puste-ale-gotowe.

**Uwaga:** ten etap nie wymaga pełnego przebiegu "agent team" z code-review — to config, nie logika. Wystarczy: Planner (mini-plan, może być bez formalnego writing-plans jeśli trywialny) → Implementer → commit.

---

## Etap 1 — Ingest pipeline + baza (Weekend 1)

**Cel:** wrzucam plik audio, dostaję przeszukiwalne chunki w bazie.

**Zakres:**
- FastAPI + PostgreSQL + pgvector przez Docker Compose (rozszerzenie Etapu 0)
- Endpoint upload pliku audio + endpoint pobrania audio z URL
- Transkrypcja Whisperem z `word_timestamps=True` — bez tego nie ma cytowań, to wymaganie twarde
- Chunking transkryptu **po naturalnych granicach wypowiedzi** (pauzy z Whispera), z overlapem, z zachowanym `start_ts`/`end_ts` na chunk — **nie chunkować po liczbie znaków**, bo chunk przecięty w połowie zdania rujnuje retrieval
- Generowanie embeddingów i zapis do pgvector
- Endpoint `GET /search?q=...` zwracający chunki + timestampy + similarity score

**Definition of Done:** upload realnego pliku audio (np. 20-30 min nagranie) kończy się kompletem chunków w bazie; `curl` do `/search` z pytaniem tematycznym zwraca trafne chunki z poprawnymi timestampami (zweryfikowane ręcznie — odsłuchaj fragment pod zwróconym timestampem, sprawdź czy się zgadza).

**Pułapka:** chunking po znakach/tokenach zamiast po granicach wypowiedzi. Test regresyjny: chunk nie powinien zaczynać/kończyć się w środku zdania (heurystyka: sprawdzić czy granice chunku pokrywają się z pauzami >0.5s w danych z Whispera).

**Do CLAUDE.md (fragment etapu):** komenda uruchomienia stacku (`docker compose up`), schema bazy (tabela chunks: id, episode_id, start_ts, end_ts, text, embedding), komenda testowego uploadu.

---

## Etap 2 — Hybrid search + reranking (Weekend 2)

**Cel:** retrieval, który działa na realnych zapytaniach, nie tylko na demo.

**Zakres:**
- BM25 przez PostgreSQL `tsvector`/`ts_rank` obok wyszukiwania semantycznego
- Fuzja wyników — Reciprocal Rank Fusion (RRF)
- Reranking — Cohere Rerank albo lokalny cross-encoder (do decyzji w Etapie 2 brainstorm: zależność od budżetu API vs. latencji lokalnej)
- Skrypt porównawczy: sam semantic vs sam BM25 vs hybrid vs hybrid+rerank na tym samym zbiorze zapytań testowych

**Definition of Done:** endpoint `/search` obsługuje parametr trybu (`semantic`/`bm25`/`hybrid`/`hybrid+rerank`); skrypt porównawczy generuje tabelę z wynikami (precision@k na ręcznie ocenionym zbiorze ~10-15 zapytań) zapisaną do pliku w `docs/`.

**Pułapka:** RRF źle zaimplementowany (np. sumowanie surowych score'ów z różnych skal zamiast rang) daje wyniki gorsze niż pojedyncza metoda — musi być test porównawczy, nie "wygląda ok".

**Do CLAUDE.md:** jak przełączać tryb wyszukiwania, gdzie leży skrypt porównawczy.

**Do notatek rozmowy kwalifikacyjnej (zapisać przy okazji tego etapu w `DECISIONS.md`):** konkretne obserwacje — gdzie semantic wygrywał, gdzie BM25, ile dał reranking (liczby, nie wrażenia).

---

## Etap 3 — LangGraph Agent (Weekend 3)

**Cel:** agent, który decyduje co zrobić, nie chain z ustaloną z góry sekwencją kroków.

**Zakres:**
- Graf ze stanem i narzędziami: `search_transcripts`, `get_context_around_timestamp`, `compare_across_episodes`, `summarise_segment`
- Conditional edges — agent decyduje czy ma dość informacji czy szukać dalej
- Checkpointing (`MemorySaver`) — historia rozmowy
- Limit kroków (guard przeciw pętli)

**Definition of Done:** test integracyjny z pytaniem wymagającym **wielu kroków** (np. porównanie dwóch odcinków) pokazuje w logu grafu więcej niż jedno wywołanie narzędzia i realną decyzję routingu; test z pytaniem prostym pokazuje jedno wywołanie — czyli ta sama architektura obsługuje oba przypadki bez hardkodowania ścieżki.

**Pułapka:** zbudowanie chaina udającego agenta — jeśli przepływ jest zawsze taki sam niezależnie od pytania, to nie jest agent. Kryterium akceptacji w kroku 5 (weryfikacja) musi jawnie pokazać różną liczbę kroków dla różnych pytań.

**Do CLAUDE.md:** lista narzędzi agenta z sygnaturami, jak dodać nowe narzędzie, gdzie jest limit kroków.

---

## Etap 4 — Warstwa głosowa (Weekend 4)

**Cel:** mówię do przeglądarki, słyszę odpowiedź, mogę kliknąć w cytowanie i skoczyć do momentu w nagraniu.

**Zakres:**
- Next.js frontend: nagrywanie z mikrofonu (MediaRecorder / Web Audio API)
- Audio → Whisper STT → tekst pytania
- Agent (Etap 3) przetwarza → odpowiedź strumieniowana do UI
- Równolegle ElevenLabs TTS → odtwarzanie odpowiedzi
- Odtwarzacz źródłowego nagrania z jumpem do cytowanego timestampu

**Definition of Done:** end-to-end demo w przeglądarce: nagranie pytania głosem → odpowiedź słyszalna → kliknięcie cytowania przewija odtwarzacz do właściwej sekundy nagrania źródłowego (zweryfikowane ręcznie w przeglądarce, patrz `run` skill).

**Pułapka:** latencja — 8-sekundowa cisza w rozmowie głosowej jest bezużyteczna. Streamować STT/LLM/TTS gdzie się da; DoD powinno zawierać zmierzony czas do pierwszego dźwięku odpowiedzi (target: zapisany w `DECISIONS.md`, np. <3s).

**Do CLAUDE.md:** jak uruchomić frontend + backend razem, zmienne środowiskowe dla kluczy API, znany limit latencji.

---

## Etap 5 — Eval harness ⭐ (Weekend 5)

**Cel:** zmierzyć jakość, nie zakładać jej. To etap o najwyższej dźwigni rekrutacyjnej — nie skracać.

**Zakres:**
- Golden dataset: 30-50 par pytanie/oczekiwana odpowiedź (source of truth: biblioteka nagrań z etapów 1-4)
- RAG triad: Context Relevance, Groundedness, Answer Relevance
- LLM-as-judge do automatycznej oceny
- Skrypt `evaluate.py` generujący raport
- Uruchomienie evala na każdej konfiguracji retrievalu z Etapu 2 (semantic/BM25/hybrid/hybrid+rerank) z konkretnymi liczbami

**Definition of Done:** `python evaluate.py` produkuje raport (JSON/markdown) z liczbami RAG triad per konfiguracja retrievalu; różnica między konfiguracjami jest widoczna i wytłumaczalna (np. "hybrid+rerank poprawił groundedness z X do Y bo...").

**Pułapka:** golden dataset zbyt mały albo złożony wyłącznie z łatwych pytań "z jednego zdania" — nie pokaże różnic między konfiguracjami. Włączyć pytania wymagające agenta z Etapu 3 (multi-hop), nie tylko prostego retrievalu.

**Do CLAUDE.md:** jak uruchomić eval, gdzie jest golden dataset, jak dodać nowy przypadek testowy.

---

## Etap 6 — MCP Server ⭐ (Weekend 6)

**Cel:** wystawić bazę wiedzy jako narzędzia MCP dla Claude Desktop / Claude Code.

**Zakres:**
- `search_knowledge_base(query)` — przeszukanie transkryptów
- `get_episode_summary(id)` — streszczenie nagrania
- `find_mentions(topic)` — gdzie w bibliotece pada dany temat
- Konfiguracja serwera do podłączenia w Claude Desktop (manifest/`claude_desktop_config.json` instrukcja w README)

**Definition of Done:** w Claude Desktop, po dodaniu serwera, zapytanie w stylu "co [gość podcastu] mówił o [temat]?" zwraca trafną odpowiedź z odwołaniem do konkretnego nagrania — nagrane jako GIF do README.

**Pułapka:** narzędzia MCP zwracające surowe, nieprzycięte dane (całe transkrypty zamiast chunków) — zapychają kontekst i dają gorsze odpowiedzi niż bezpośrednie query do `/search`. Trzymać kontrakt narzędzia zwięzły (chunk + timestamp + episode, nie cały transkrypt).

**Do CLAUDE.md:** lista narzędzi MCP, jak podłączyć serwer lokalnie, przykładowe zapytania.

---

## Etap 7 — Deploy, README, polish (Weekend 7)

**Cel:** projekt gotowy do pokazania rekruterowi bez tłumaczenia się.

**Zakres:**
- Deploy: Vercel (frontend) + Railway lub Render (FastAPI + Postgres) — albo świadoma decyzja "tylko Docker Compose + wideo" jeśli czasu zabrakło (patrz priorytety na górze dokumentu)
- `docker-compose.yml` finalny, do uruchomienia lokalnego jedną komendą
- `README.md` w strukturze: (1) jedno zdanie co to robi, (2) GIF demo głosowe, (3) diagram architektury, (4) sekcja **Evaluation** z tabelą/wykresem z Etapu 5 wysoko w dokumencie, (5) decyzje techniczne i uzasadnienia (z `DECISIONS.md`), (6) MCP server + GIF z Etapu 6, (7) setup lokalny, (8) znane ograniczenia
- **Kompilacja `CLAUDE.md`**: scalenie wszystkich `docs/claude-md-fragments/etap-*.md` w jeden finalny `CLAUDE.md` w root (komendy, architektura, konwencje, gotchas) — to jest ten plik, o którym mówiłeś że przygotujesz go dla VSC; tutaj następuje jego faktyczne złożenie z materiału zebranego etapami

**Definition of Done:** świeży `git clone` + `docker compose up` + komenda z README uruchamia cały stack lokalnie bez ręcznych poprawek; README czytany od góry do dołu tłumaczy projekt bez dodatkowych pytań; `CLAUDE.md` istnieje i pozwala nowej sesji Claude Code zacząć pracować w repo bez dodatkowego kontekstu od Ciebie.

**Do CLAUDE.md:** ten etap SAM JEST kompilacją CLAUDE.md — więc efekt tego etapu to plik końcowy, nie kolejny fragment.

---

## Etap 15 — Streaming voice: WebSocket duplex, barge-in, zmierzona latencja

> Dodane po Etapie 14 (produkcyjny deploy) jako rozszerzenie roadmapy — domyka lukę, która jest bezpośrednim powodem istnienia tego projektu wg sekcji `Kontekst i uzasadnienie`: "Voice AI Engineer, który rozumie latencję konwersacji, VAD, streaming STT". Zweryfikowane researchem konkurencji (Vapi, Retell AI, Bland AI) w trakcie brainstormingu tego etapu: latencja end-to-end (~700-900ms) i obsługa barge-in to główny argument produktowy tych platform, nie dodatek. Domyka też DoD Etapu 4, które nigdy nie zostało w pełni spełnione (zmierzony czas do pierwszego dźwięku nigdy nie trafił do `DECISIONS.md` z liczbami).

**Cel:** rozmowa głosowa z agentem działa jak duplex, nie jak walkie-talkie — mówisz, agent może cię przerwać albo ty jego, i wiadomo dokładnie ile to trwa.

**Kontekst (stan przed etapem):** obecny flow to `AudioRecorder.tsx` (nagrywanie z energy-based VAD auto-stop) → cały blob → `POST /voice/stt` (Whisper, batch, jeden strzał) → `sendMessage()` → `POST /agent/chat/stream` (SSE: tokeny tekstu + audio per zdanie z `SentenceChunker`) → sekwencyjne odtwarzanie w `AudioContext`. To jest streaming *odpowiedzi*, ale nie *rozmowy* — mikrofon nie słucha podczas odtwarzania, nie ma przerywania, nie ma pomiaru latencji nigdzie w kodzie.

**Zakres:**
- Nowy provider: **Deepgram streaming STT** (`DEEPGRAM_API_KEY`) — prawdziwe interim transcripts słowo-po-słowie, nie batch. Whisper zostaje bez zmian dla ingestu (word-level timestamps do chunkingu już działają dobrze, nie ma powodu ruszać).
- Backend: nowy endpoint `WS /voice/stream` (`backend/app/voice/stream_ws.py`) — klient streamuje audio ciągle (nie jeden blob), backend przekazuje do Deepgram, na sygnale końca wypowiedzi (Deepgram endpointing) odpala istniejący graf LangGraph i strumieniuje odpowiedź (tekst + TTS per zdanie, reużywając `SentenceChunker` i `voice/tts.py`) z powrotem przez to samo WS.
- Cienki wrapper `backend/app/voice/deepgram_client.py` do streaming API Deepgrama (websockets/httpx, wzorem istniejących lazy-client patternów z `embedder.py`/`reranker.py`).
- **Barge-in:** mikrofon zostaje aktywny podczas odtwarzania TTS. Gdy Deepgram zgłosi nową mowę w trakcie playbacku, klient natychmiast czyści kolejkę `AudioContext` i wysyła sygnał anulowania przez WS; backend najlepszym wysiłkiem przerywa zadania TTS w locie (bez gwarancji idealnej synchronizacji — porzucony chunk audio po stronie klienta wystarczy).
- **Pomiar latencji:** klient znakuje czasem koniec wypowiedzi użytkownika i moment pierwszego bajtu audio odpowiedzi; różnica pokazywana jako mały badge przy odpowiedzi ("⚡ 640ms"); logowana strukturalnie po stronie backendu, żeby `DECISIONS.md` dostał realne liczby p50/p95, nie deklarację celu.
- **Fallback:** gdy Deepgram niedostępny/błąd — automatyczny powrót do istniejącej ścieżki `/voice/stt` + `/agent/chat/stream` (obecne zachowanie zostaje jako degradacja, nie znika).
- Frontend: nowy `frontend/components/VoiceStream.tsx` (zastępuje ścieżkę nagrywania w `AudioRecorder.tsx` dla trybu duplex, stary komponent zostaje jako fallback UI) + `frontend/lib/voiceWebSocket.ts` (reconnect z backoff).

**Nie-cele (poza zakresem tego etapu):** telefonia (Twilio/SIP), przełączanie języka głosu w locie, wake-word detection, zmiana STT dla ingestu.

**Definition of Done:** end-to-end demo w przeglądarce pokazuje (1) odpowiedź agenta przerwaną w połowie zdania przez nową wypowiedź użytkownika, ze słyszalnym, natychmiastowym urwaniem playbacku; (2) badge z realnym czasem do pierwszego dźwięku przy każdej odpowiedzi; (3) symulowana awaria Deepgrama (zły klucz) skutkuje płynnym powrotem do starej ścieżki `/voice/stt`, nie błędem w UI; (4) `DECISIONS.md` ma wpis z realnymi liczbami p50/p95 latencji zmierzonymi na kilkunastu próbkach, nie deklaracją "powinno być <3s".

**Pułapka:** ten sam typ buga co już raz uderzył w tym projekcie (Etap 15 wcześniejszy, streaming SSE) — kształt danych ze streamingowego API bywa inny niż się wydaje z dokumentacji (por. `_extract_text` dla Anthropica, gdzie `content` okazał się listą bloków, nie stringiem). Zweryfikować realny kształt eventów z Deepgram WS na żywo, nie zakładać ze specyfikacji, i pokryć to testem regresyjnym analogicznym do `test_ignores_tool_use_blocks_interleaved_with_text`. Kolejność operacji (zapis/anulowanie vs. TTS) też już raz ugryzła (historia czatu zapisywana po TTS zamiast przed) — pilnować kolejności: potwierdzenie że tekst dotarł do klienta ma priorytet nad domykaniem audio.

**Do CLAUDE.md:** nowy endpoint WS i jego kontrakt eventów, `DEEPGRAM_API_KEY` w tabeli zmiennych środowiskowych, jak działa fallback, gdzie szukać logów latencji.

---

## Etap 16 — Diaryzacja mówców + auto-summary/chapters per odcinek

> Domyka realną lukę featerową wobec najlepszych narzędzi w niszy podcast-AI (Snipd, NotebookLM) zidentyfikowaną researchem konkurencji przy brainstormingu tego etapu, zostając w temacie audio ML zamiast uciekać w generyczny product feature. Celowo zsekwencjonowane po Etapie 15, bo reużywa tego samego konta Deepgram (diaryzacja to `diarize=true` w ich prerecorded API) — jeden nowy provider obsługuje oba etapy, nie dwa.

**Cel:** cytowanie mówi "Patrick O'Neill powiedział..." zamiast anonimowego fragmentu tekstu, a odcinek ma streszczenie i spis rozdziałów zanim ktokolwiek zada pierwsze pytanie.

**Zakres:**
- Baza: nowa kolumna `Chunk.speaker_label` (nullable String), `Episode.summary` (nullable Text), `Episode.chapters` (nullable JSON — ten sam wzorzec co już istniejące `ChatMessage.sources`, bez nowej tabeli na start).
- `backend/app/ingest/diarizer.py` (nowy): wywołanie Deepgram prerecorded API z `diarize=true` na tym samym pliku audio co transkrypcja; mapowanie segmentów mówców na chunki przez nakładanie się przedziałów czasowych (`[start_ts, end_ts]` chunku → mówca o największym pokryciu).
- `backend/app/ingest/summarizer.py` (nowy): jedno wywołanie Claude Haiku (ten sam model co LLM-as-judge w evalu) nad zchunkowanym transkryptem, structured output produkujący: streszczenie (2-4 zdania), chapter markers (`[{start_ts, title}]`), best-effort mapowanie Speaker N → prawdziwe imię jeśli pada w transkrypcie (np. "jestem Gary Jordan, wasz gospodarz") — w tym samym wywołaniu, nie osobnym.
- Integracja w `ingest_audio()`/`ingest_from_url()`/`ingest_from_rss()`: diaryzacja + summarization uruchamiane synchronicznie po chunkingu i embeddingu, przed zwróceniem odpowiedzi ingestu (ingest już dziś bywa wieloczynnościowy — UI ma copy "to może potrwać kilka minut").
- API: `GET /episodes/{id}` (i `EpisodeResponse` gdzie relevantne) rozszerzone o `summary`, `chapters`, `chunks[].speaker_label`.
- Frontend: `SourcePlayer.tsx` prefiksuje cytat etykietą mówcy; widok skupionego odcinka w sidebarze dostaje sekcję streszczenia i klikalny spis rozdziałów (jump do timestampu, analogicznie do istniejących source-jump linków).

**Nie-cele:** rozpoznawanie tego samego głosu między różnymi odcinkami (voice fingerprinting), ręczna edycja etykiet mówców przez użytkownika.

**Definition of Done:** zaingestowany odcinek z co najmniej dwoma mówcami pokazuje w UI poprawnie rozdzielone etykiety przy cytatach (zweryfikowane ręcznie ze słuchem — odsłuchaj fragment, sprawdź czy etykieta się zgadza); strona odcinka pokazuje streszczenie i klikalne rozdziały, kliknięcie skacze do właściwego miejsca; symulowana awaria diaryzacji/summaryzacji (zły klucz/timeout) kończy się odcinkiem z `speaker_label`/`summary` pustym, ale w pełni przeszukiwalnym — ingest się nie wywala.

**Pułapka:** mapowanie mówca→chunk przez nakładanie przedziałów czasowych może dawać niejednoznaczne wyniki na granicach wypowiedzi (dwóch mówców nakładających się, przerywających sobie) — potrzebny jawny próg (np. mówca musi pokrywać >50% długości chunku, inaczej `speaker_label = NULL` zamiast zgadywać). Structured output z LLM dla chapters/summary bywa niestabilny formatem — wymusić schemat (function calling / JSON mode), nie parsować wolnego tekstu regexem.

**Do CLAUDE.md:** nowe kolumny w `Chunk`/`Episode`, jak działa mapowanie mówca→chunk, kontrakt structured output summaryzatora, gdzie w UI pojawia się streszczenie/rozdziały.

---

## Materiał na rozmowę kwalifikacyjną (zbierany po drodze, nie osobny etap)

Zbieraj to na bieżąco w `DECISIONS.md`, nie odtwarzaj z pamięci przed rozmową:
- **"Opowiedz o projekcie"** → zacznij od problemu (audio nieprzeszukiwalne jak tekst), nie od stacku.
- **"Jak mierzysz czy działa?"** → Etap 5: golden dataset, RAG triad, konkretne liczby, porównanie konfiguracji.
- **"Dlaczego LangGraph a nie chain?"** → Etap 3: liczba kroków zależy od pytania, DoD tego etapu to właśnie demonstruje.
- **"Największe problemy?"** → latencja (Etap 4) i chunking z zachowaniem timestampów (Etap 1) — z konkretnym rozwiązaniem, nie "wszystko poszło gładko".

**Wpis do CV po zakończeniu (do `README.md` i CV równolegle):**
> Voice Knowledge Agent — open-source agent głosowy nad biblioteką transkrybowanych nagrań. LangGraph z wieloma narzędziami, hybrid search (semantic + BM25) z rerankingiem na pgvector, warstwa głosowa Whisper + ElevenLabs, własny MCP server, oraz automatyczny eval harness (RAG triad, LLM-as-judge) mierzący jakość retrievalu i groundedness. `Python · FastAPI · LangGraph · pgvector · Whisper · ElevenLabs · MCP · Next.js`

---

## Uwaga o tempie

Claude Code przyspiesza pisanie kodu, nie zrozumienie. Nikt na rozmowie nie zapyta jak szybko to powstało — zapytają dlaczego wybrałeś dany chunking i co się stało gdy spróbowałeś innego. Krok 7 każdego etapu (`DECISIONS.md`) istnieje właśnie po to, żeby to wymusić — 2 minuty na wpis, zero pracy przed rozmową. Jeśli krok 7 zacznie się pomijać "bo oczywiste", to pierwszy sygnał że etapy lecą na autopilocie.
