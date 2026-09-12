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

## [Etap 6] MCP tools jako httpx wrappers zamiast direct DB access (2026-09-05)

**Decyzja:** narzedzia MCP (`tools.py`) wywoluja backend REST API przez httpx zamiast bezposrednio laczyc sie z baza danych.

**Alternatywy odrzucone:** bezposrednie importy z `backend/app` (SQLAlchemy session); osobna kopia session/models w mcp-server.

**Uzasadnienie:** MCP server jest osobnym procesem — duplikowanie logiki DB byloby naruszeniem DRY i wymagaloby osobnych migracji. Wywolanie HTTP gwarantuje te same transformacje co backend (RRF, reranking, walidacja) i pozwala na niezalezne skalowanie. Jedyna wada: dodatkowe opoznienie HTTP; przy lokalnym deploymencie pomijalne (<5ms).

---

## [Etap 6] Osobny venv dla mcp-server z powodu konfliktu starlette (2026-09-05)

**Decyzja:** `mcp-server/` ma wlasny `requirements.txt` i wymaga osobnego venv. Testy narzedzi (`test_mcp_tools.py`) uruchamiane w venv backendu (mcp package nie potrzebny do testow tools.py).

**Alternatywy odrzucone:** upgrade fastapi do wersji kompatybilnej z mcp; pinning mcp do starszej wersji.

**Uzasadnienie:** `mcp>=1.0` wymaga `starlette>=1.0.0` (przez `sse-starlette`); `fastapi 0.115.0` wymaga `starlette<0.39.0`. Konflikt nierozwiazalny w jednym venv bez upgrade fastapi (ktory moglby zepsuc inne zaleznosci). Rozdzielenie venvow jest naturalnym rozwiazaniem — MCP server jest niezaleznym procesem, nie biblioteka backendu.

---

## [Etap 5] LLM-as-judge zamiast referencyjnych odpowiedzi w golden dataset (2026-09-05)

**Decyzja:** golden dataset zawiera `expected_topics: [str]` zamiast gotowych "ground-truth" odpowiedzi; ocena (Context Relevance, Groundedness, Answer Relevance) wykonywana jest przez LLM-sedziego (Anthropic Haiku).

**Alternatywy odrzucone:** manualne odpowiedzi reference i porownanie string-similarity (BLEU/ROUGE); zwykly recall na expected_topics bez LLM.

**Uzasadnienie:** Nie mamy prawdziwych podcastow — generowanie referencyjnych odpowiedzi do syntetycznych pytan byloby circular. LLM-as-judge mierzy to samo co interesuje nas jako uzytkownika: czy odpowiedz jest trafna i ugruntowana w kontekscie. Haiku zamiast Sonnet/Opus: latencja i koszt; przy 15 pytan x 4 tryby x 3 metryki = 180 wywolan; Haiku wystarczy do oceny.

---

## [Etap 5] Skalowanie scoru 1-5 do 0-1 zamiast bezposredniego float (2026-09-05)

**Decyzja:** prompt prosi LLM o liczbe calkowita 1-5; `_parse_score` normalizuje do `(val-1)/4.0`. Parsowanie wyciaga pierwszy cyfre ze stringa (zabezpieczenie na "The score is 4." etc.).

**Alternatywy odrzucone:** proszenie o float bezposrednio (0.0-1.0); prompt z rubrykami i calkowitym JSON.

**Uzasadnienie:** Male modele (Haiku) sa bardziej deterministyczne w generowaniu pojedynczej cyfry niz floata. Parse pierwszej cyfry ze stringa jest odporne na dodatkowy tekst ktory maly model moze dorzucic. JSON parsing nadklada i tak nie uzasadniona dodatkowa zlozonosc dla tak prostego wyjscia.

---

## [Etap 4] ElevenLabs SDK zastapiony bezposrednim wywolaniem httpx (2026-09-05)

**Decyzja:** TTS nie korzysta z oficjalnego SDK (`elevenlabs>=1`), lecz wywoluje REST API ElevenLabs bezposrednio przez `httpx.post`.

**Alternatywy odrzucone:** `elevenlabs>=1` (oficjalny SDK Python); `httpx` w osobnym wrapperze biblioteki.

**Uzasadnienie:** Instalacja SDK na Windows 11 konczy sie bledem `OSError: [Errno 2] No such file or directory` — SDK tworzy pliki pomocnicze o sciezkach dluzszych niz 260 znakow (limit systemu Windows bez wlaczonej opcji LongPathsEnabled). Bezposrednie wywolanie REST API jest rownowazne funkcjonalnie, eliminuje zewnetrzna zaleznosc i jest zgodne z `httpx` juz obecnym w requirements.txt od Etapu 1.

---

## [Etap 4] Whisper STT zwraca plain string, nie segmenty (2026-09-05)

**Decyzja:** `voice/stt.py::transcribe_audio` zwraca `str` (caly tekst). Pipeline ingestowy (`ingest/transcriber.py`) zwraca liste segmentow z timestampami.

**Alternatywy odrzucone:** wspolny format zwracania segmentow dla obu sciezek; zwracanie timestampow dla zapytan glosowych.

**Uzasadnienie:** Zapytania glosowe nie wymagaja timestampow — uzytkownik mowi pytanie (kilka sekund), a nie caly podcast. Utrzymywanie dwoch odrebnych sygnatury jest tu wlasciwe: ingest potrzebuje granic segmentow do chunk-owania; voice/STT potrzebuje tylko tekstu do przekazania agentowi.

---

## [Etap 1] Konwencja: 1 squash-commit per etap na `main` (2026-09-05)

**Decyzja:** granularne commity (per-task) tworzone sa na branchu roboczym podczas developmentu i code-review; przed scaleniem z `main` caly branch jest sciskany do jednego commita z pelnym opisem zmian.

**Alternatywy odrzucone:** scalanie wszystkich granularnych commitow do `main`; squash-merge przez GitHub.

**Uzasadnienie:** `git log` na `main` ma odzwierciedlac strukture etapow projektu — jeden commit = jeden etap = jedna sposta zmiana. Granularne commity sa dostepne w historii brancha do czasu jego usuniecia, co wystarcza do code-review i debugowania podczas trwania etapu.

---

## [Etap 9] Server-Sent Events (SSE) zamiast WebSocket dla streamingu odpowiedzi (2026-09-05)

**Decyzja:** `POST /agent/chat/stream` zwraca `StreamingResponse` z `media_type="text/event-stream"` (SSE) zamiast otwierac polaczenie WebSocket.

**Alternatywy odrzucone:** WebSocket (`fastapi.WebSocket`); dlugie pollowanie (`long polling`).

**Uzasadnienie:** Komunikacja jest jednokierunkowa (serwer → klient, jeden request → jeden strumien odpowiedzi) — WebSocket dodalby zlozonosc dwukierunkowego protokolu bez realnej korzysci. SSE dziala na zwyklym `fetch` + `ReadableStream` po stronie przegladarki (bez dodatkowej biblioteki), automatycznie zamyka polaczenie po `done`, i przechodzi przez istniejacy Next.js rewrite proxy (`/api/:path*`) bez zmian w konfiguracji.

---

## [Etap 9] Chunkowanie po granicach zdan (SentenceChunker) dla TTS w locie (2026-09-05)

**Decyzja:** `backend/app/agent/chunker.py::SentenceChunker` buforuje strumieniowane tokeny LLM i emituje fragment do syntezy mowy dopiero po osiagnieciu granicy zdania (`[.!?]\s+`) ORAZ minimum 8 slow (`MIN_WORDS`), scalajac krotkie zdania wiodace (np. "Tak.") z kolejnymi, zamiast wysylac je do TTS osobno.

**Alternatywy odrzucone:** synteza calej odpowiedzi na koniec (brak streamingu audio — pierwotne podejscie z Etapu 4); chunkowanie po stalej liczbie znakow/tokenow bez wzgledu na granice zdan.

**Uzasadnienie:** Wysylanie kazdego pojedynczego krotkiego zdania do ElevenLabs osobno psuje naturalnosc audio (urwane, zbyt czeste fragmenty) i mnozy liczbe platnych wywolan API. Próg 8-slowny gwarantuje, ze kazdy fragment audio brzmi jak pelna, sensowna fraza, a zdania krotsze sa doklejane do nastepnych zamiast tworzyc osobny (zbyt krotki) request do TTS.

---

## [Etap 9] Rownolegle wywolania TTS, ale odtwarzanie w kolejnosci powstania (2026-09-05)

**Decyzja:** `stream_agent_response` planuje kazdy fragment zdania jako niezalezny `asyncio.create_task` (rownolegle wywolania ElevenLabs), ale `await`-uje je w kolejnosci utworzenia przed wyslaniem zdarzenia `audio` przez SSE — gwarantuje to poprawna kolejnosc odtwarzania mimo rownoleglej syntezy.

**Alternatywy odrzucone:** sekwencyjne `await` kazdego wywolania TTS przed rozpoczeciem kolejnego (prostsze, ale sumuje latencje kazdego wywolania API zamiast je nakladac); wysylanie audio w kolejnosci ukonczenia (`asyncio.as_completed`) — szybsze, ale psuje kolejnosc odtwarzania po stronie przegladarki.

**Uzasadnienie:** `await task[i]` blokuje tylko do ukonczenia zadania `i`, niezaleznie od tego czy zadanie `i+1` skonczylo sie wczesniej — dzieki temu kolejnosc odtwarzania = kolejnosc zdan w odpowiedzi, a czas oczekiwania na cala odpowiedz jest ograniczony przez najwolniejsze wywolanie, nie przez ich sume. Generator dodatkowo obejmuje cala petle `try/except/finally` — blad w dowolnym miejscu (LLM lub TTS) wysyla zdarzenie `error` zamiast ubijac strumien bez wyjasnienia, a niedokonczone zadania TTS sa anulowane przy przedwczesnym zamknieciu polaczenia (np. klient rozlacza sie w trakcie), zeby nie generowac platnych zapytan do API dla nikogo.

---

## [Etap 12] Prosty energy-based VAD zamiast Silero/WebRTC VAD (2026-09-05)

**Decyzja:** `frontend/components/AudioRecorder.tsx` wykrywa koniec wypowiedzi przez prog amplitudy RMS liczony z `AnalyserNode.getByteTimeDomainData` (Web Audio API), bez zadnej dodatkowej biblioteki — nagrywanie zatrzymuje sie automatycznie po 1.5s ciszy, ale dopiero gdy wczesniej wykryto mowe (min. 300ms powyzej progu).

**Alternatywy odrzucone:** Silero VAD przez `@ricky0123/vad-web` (onnxruntime-web + pliki modelu ONNX doladowywane w przegladarce); natywny WebRTC VAD (biblioteka C, brak dojrzalego bindingu przegladarkowego).

**Uzasadnienie:** Silero VAD dodaje ciezka zaleznosc (WASM runtime + kilkumegabajtowy model) i komplikuje bundling w Next.js/Docker na Windows — nieproporcjonalny koszt jak na funkcje "polish", nie rdzennie techniczna (w przeciwienstwie do streamingu z Etapu 9). Prog RMS jest w pelni wystarczajacy dla typowego przypadku uzycia (krotkie pytanie glosowe w cichym otoczeniu) i nie wymaga zadnej nowej zaleznosci npm — cala logika miesci sie w jednym pliku komponentu.

---

## [Etap 12] `feedparser` zamiast recznego parsowania XML dla RSS ingest (2026-09-05)

**Decyzja:** `backend/app/ingest/rss.py::parse_feed` korzysta z biblioteki `feedparser` (czysty Python, bez zaleznosci C) do parsowania feedow RSS/Atom i wyciagania enclosure z audio.

**Alternatywy odrzucone:** recznе parsowanie przez `xml.etree.ElementTree` z wlasna logika obslugi RSS 2.0 / Atom / itunes namespace; `lxml` (wymaga kompilacji C, ryzyko problemow na Windows analogiczne do `elevenlabs>=1` z Etapu 4).

**Uzasadnienie:** `feedparser` jest de facto standardem do parsowania podcastowych feedow w Pythonie — normalizuje roznice miedzy RSS 2.0 i Atom, obsluguje malformed XML (tryb "bozo") i nie wymaga kompilacji natywnej (czysty Python), co jest istotne po doswiadczeniu z `elevenlabs` SDK na Windows w Etapie 4. Batch-owe niepowodzenie calego ingest (np. nieosiagalny URL feedu) jest obslugiwane w `POST /ingest/rss` przez `HTTPException(400, ...)`, spojnie z `/ingest/upload` i `/ingest/url`; niepowodzenie pojedynczego odcinka w ramach batcha jest laczone w liste `errors` zamiast przerywac cala operacje.

---

## [Etap 13] Ochrona SSRF: walidacja schematu + klasyfikacja adresu IP zamiast pelnego pinningu polaczenia (2026-09-06)

**Decyzja:** `backend/app/core/url_safety.py::assert_safe_url` odrzuca URL-e z innym schematem niz http/https oraz takie, ktorych hostname rozwiazuje sie (przez `socket.getaddrinfo`) na adres prywatny/loopback/link-local/reserved/multicast/unspecified (w tym warianty IPv4-mapped IPv6). Kazdy redirect w kliencie `httpx` jest ponownie walidowany przez `event_hooks={"request": [...]}`. Wywolywana w `ingest_from_url` i `parse_feed` przed jakimkolwiek realnym polaczeniem sieciowym.

**Alternatywy odrzucone:** brak walidacji (stan sprzed Etapu 13, podatny na SSRF do metadanych chmury/uslug wewnetrznych); pelne pinowanie polaczenia do zweryfikowanego adresu IP przez wlasny transport HTTP — zamknieloby luke DNS-rebinding (TOCTOU miedzy sprawdzeniem adresu a faktycznym polaczeniem httpx), ale wymagaloby wlasnej implementacji warstwy transportowej.

**Uzasadnienie:** Endpointy `/ingest/url` i `/ingest/rss` pozwalaly serwerowi pobrac dowolny URL podany przez uzytkownika. Walidacja na poziomie DNS + klasyfikacji IP zatrzymuje realistyczne ataki (`file://`, `localhost`, `169.254.169.254`, sieci prywatne) przy minimalnym koszcie implementacyjnym i zerowych nowych zaleznosciach. Ryzyko DNS-rebinding pozostaje swiadomie zaakceptowanym ograniczeniem — nieproporcjonalne dla projektu portfolio bez wlasnej infrastruktury transportowej; oba endpointy sa dodatkowo za opcjonalna bramka `X-API-Key` i rate-limitem, co znaczaco podnosi koszt takiego ataku.

---

## [Etap 13] Ograniczenie i uwierzytelnianie tylko dla `/ingest/*`, nie dla czatu/wyszukiwania/glosu (2026-09-06)

**Decyzja:** Router `/ingest` ma `dependencies=[Depends(require_api_key)]` (opcjonalny wspoldzielony klucz przez naglowek `X-API-Key`, aktywny tylko gdy zmienna `API_KEY` jest ustawiona) oraz rate-limity `slowapi` (5-10/min). `/agent/chat`, `/agent/chat/stream`, `/voice/stt`, `/voice/tts` maja rate-limity (20/min), ale nie wymagaja klucza API.

**Alternatywy odrzucone:** wymog klucza API na wszystkich endpointach (w tym czacie/wyszukiwaniu) — zmienialby charakter portfolio-demo z "sprobuj live" na "tylko dla wlasciciela"; pelny system logowania uzytkownikow (nieproporcjonalny do zakresu projektu single-user).

**Uzasadnienie:** Endpointy ingest sa operacjami zapisu generujacymi koszt (Whisper) niezaleznie od tego, czy dane sa potem uzywane — sensowne jest ograniczenie ich do operatora. Czat/wyszukiwanie/glos sa rdzeniem demo, ktore ma dzialac od reki dla kazdego odwiedzajacego; sam rate-limiting jest tu wystarczajacym zabezpieczeniem kosztowym. `API_KEY` jest domyslnie puste (brak wymogu) — swiadomy kompromis udokumentowany w `.env.example`: operator powinien je ustawic przed produkcyjnym udostepnieniem, jesli chce ograniczyc kto moze dorzucac nowe nagrania. Frontend wstrzykuje naglowek `X-API-Key` po stronie serwera przez `frontend/middleware.ts` (nigdy w kodzie dzialajacym w przegladarce), zeby ustawienie klucza nie psulo UI do ingestu.

---

## [Etap 13] Klucz rate-limitera: `X-Forwarded-For` zamiast surowego adresu polaczenia TCP (2026-09-06)

**Decyzja:** `backend/app/core/rate_limit.py::_client_key` odczytuje pierwszy adres z naglowka `X-Forwarded-For`, jesli jest obecny, zamiast domyslnego w `slowapi` `request.client.host`.

**Alternatywy odrzucone:** `slowapi.util.get_remote_address` bez modyfikacji (zachowanie domyslne).

**Uzasadnienie:** Caly ruch przegladarki przechodzi przez serwerowy rewrite-proxy Next.js (`/api/:path*`), wiec `request.client.host` widzialby zawsze adres kontenera frontendu, a nie prawdziwego uzytkownika — kazdy limit zbieralby sie do jednego wspoldzielonego "kubelka" dla wszystkich odwiedzajacych zamiast dzialac per-uzytkownik. Odczyt `X-Forwarded-For` (ustawianego przez wiekszosc proxy produkcyjnych, w tym docelowy Vercel z planu Etapu 11) przywraca zamierzone dzialanie limitow.

---

## [Etap 13] Sniffing magic bytes zamiast `libmagic`/`python-magic` dla walidacji uploadu (2026-09-06)

**Decyzja:** `backend/app/ingest/validators.py::is_probably_audio` sprawdza pierwsze bajty pliku pod katem znanych sygnatur kontenerow audio (ID3/MP3 frame sync, RIFF+WAVE, Ogg, ftyp/M4A, WebM/EBML) zamiast uzywac biblioteki do rozpoznawania typow MIME.

**Alternatywy odrzucone:** `python-magic` (wiaze `libmagic`, biblioteke C — ryzyko problemow na Windows analogiczne do `elevenlabs>=1` z Etapu 4 i `lxml` z Etapu 12).

**Uzasadnienie:** Endpoint `/ingest/upload` przyjmowal dowolne dane binarne bez zadnej walidacji tresci, marnujac platne wywolania Whisper na oczywiscie nie-audio pliki. Prosty sniffing sygnatur pokrywa realistyczne przypadki (typowe formaty audio) bez nowej zaleznosci binarnej, spojnie z reszta projektu, ktory konsekwentnie unika bibliotek wymagajacych kompilacji C na Windows.

---

## [Etap 13] Dwupoziomowa obsluga bledow: znane wyjatki -> 400 z tresc; nieoczekiwane -> 500 generyczny + log (2026-09-06)

**Decyzja:** Kazdy router (`ingest`, `search`, `agent/chat`) rozroznia `(ValueError, RuntimeError)` (swiadomie rzucane przez wlasny kod, bezpieczna tresc) od pozostalych wyjatkow (`Exception`), ktore sa logowane po stronie serwera (`logger.exception`) i zwracane jako `HTTPException(500, "Internal error...")` bez surowej tresci.

**Alternatywy odrzucone:** pozostawienie `except Exception as e: raise HTTPException(400, str(e))` (stan sprzed Etapu 13) — przekazywalo surowa tresc kazdego wyjatku, w tym fragmenty odpowiedzi API zewnetrznych i sciezki systemowe, bezposrednio do klienta.

**Uzasadnienie:** Rozroznienie po typie wyjatku pozwala zachowac uzyteczne komunikaty dla oczekiwanych przypadkow (np. "ANTHROPIC_API_KEY not set", bledny URL rozpoznany przez `httpx.HTTPError`, blad transkrypcji z `openai.APIError`) bez utraty ich czytelnosci, jednoczesnie chroniac przed przypadkowym wyciekiem przy nieoczekiwanych awariach — te trafiaja teraz wylacznie do logow serwera. Sciezka bledow strumieniowanych przez SSE (`/agent/chat/stream`) pozostaje bez zmian (przekazuje surowa tresc bledu w zdarzeniu `error`) — swiadomie pozostawiona poza zakresem etapu, bo jedynym odbiorca tego strumienia jest ten sam klient przegladarki, ktory wyslal zapytanie.

---

## [Etap 14] Watek `sources_sink` + `mode` przez `make_tools -> build_graph -> stream_agent_response` (2026-09-06)

**Decyzja:** Agentowe narzedzia wyszukujace (`search_transcripts`, `get_context_around_timestamp`, `compare_across_episodes`) przyjmuja opcjonalny `sources_sink: list[dict]`, do ktorego dopisuja kazdy pobrany fragment transkryptu; `make_tools`/`build_graph`/`stream_agent_response` przekazuja go dalej lancuchem, a `stream_agent_response` emituje go jako jedno zdarzenie SSE `sources` (odchudzone do 4 pol UI, zdeduplikowane, ograniczone do `MAX_SOURCES = 8`) tuz przed `done`. Analogicznie przekazywany jest `mode` (ktora funkcja wyszukujaca ma uzyc agent), sterowany selectorem w UI.

**Alternatywy odrzucone:** osobny endpoint `/agent/chat/sources` odpytywany po zakonczeniu odpowiedzi (dodatkowy round-trip, ryzyko niespojnosci z tym, co faktycznie wykorzystal agent); parsowanie odpowiedzi LLM w poszukiwaniu cytowan w tekscie (kruche, zalezne od formatu wyjscia modelu).

**Uzasadnienie:** Narzedzia agenta juz pobieraly strukturalne dane (`episode_id`, `start_ts`, `end_ts`, `text`) z hybrid/BM25/semantic search, ale zwracaly LLM-owi wylacznie sformatowany string — cytowania z znacznikami czasu (headline feature projektu) nigdy nie docieraly do frontendu. Zbieranie ich w mutowalnej liscie przekazanej przez closure to najmniejsza zmiana, ktora nie wplywa na to, co widzi LLM (wciaz dostaje tylko sformatowany tekst) ani na istniejace testy (parametry opcjonalne, domyslnie `None`/`"hybrid"`). Krotki `mode` string jest przekazywany zamiast obiektu funkcji, zeby `make_tools` mogl bezpiecznie zbudowac mapowanie mode->funkcja przy kazdym wywolaniu (co zachowuje zgodnosc z istniejacymi testami mockujacymi `app.agent.tools.hybrid_search` na poziomie modulu). `hybrid+rerank` bez skonfigurowanego `COHERE_API_KEY` cicho spada do `hybrid` w tej sciezce (w przeciwienstwie do bezposredniego `GET /search`, ktory jawnie zwraca blad) — agent nie ma jak zakomunikowac uzytkownikowi bledu narzedzia bez zuzycia calego budzetu `MAX_STEPS` na bezowocne retry.

---

## [Etap 14] Odtwarzanie cytowanych fragmentow przez Media Fragments URI (`#t=start,end`) zamiast wlasnego serwowania audio (2026-09-06)

**Decyzja:** `SourcePlayer.tsx` renderuje `<audio src="{episode.source_url}#t=start,end">` gdy odcinek ma zapisany `source_url` (ingest przez URL/RSS); dla plikow wgranych bezposrednio (`source_url` puste) pokazuje tylko tekst "Audio unavailable for uploaded files."

**Alternatywy odrzucone:** trwale przechowywanie przeslanego/pobranego audio po stronie serwera i serwowanie go przez wlasny endpoint (obecnie `ingest_audio`/`ingest_from_url` usuwaja plik tymczasowy zaraz po transkrypcji — dodanie trwalego storage to osobna, wieksza funkcja, poza zakresem etapu skupionego na naprawie istniejacych elementow UI).

**Uzasadnienie:** Media Fragments URI (`#t=`) jest natywnie wspierany przez `<audio>` w Chrome/Firefox bez zadnego JS do seekowania — najtanszy sposob na realne odtwarzanie cytowanego fragmentu bez budowania nowej infrastruktury. Ograniczenie do URL/RSS-owych odcinkow jest uczciwym kompromisem: to one i tak maja zewnetrznie hostowany plik audio, wiec nie wymaga to zadnego nowego przechowywania danych po stronie serwera. Trwale przechowywanie audio z uploadu zostaje odnotowane jako zaleglosc na przyszly etap, jesli pelna playback-dla-wszystkich-zrodel bedzie potrzebna.

---

## [Etap 14] Naprawa `NEXT_PUBLIC_API_URL`: nazwa uslugi compose (`backend`) zamiast `localhost` (2026-09-06)

**Decyzja:** `docker-compose.yml`'s frontend service ustawia `NEXT_PUBLIC_API_URL=http://backend:8000` zamiast `http://localhost:8000`.

**Alternatywy odrzucone:** pozostawienie `localhost:8000` (stan sprzed Etapu 14, w rzeczywistosci niedzialajacy).

**Uzasadnienie:** Proxy Next.js (`rewrites()` w `next.config.ts`) wykonuje sie po stronie serwera, wewnatrz kontenera `frontend` — `localhost` w tym kontekscie oznacza sam kontener, na ktorym nic nie nasluchuje na porcie 8000, wiec kazde wywolanie `/api/*` z prawdziwej przegladarki konczylo sie bledem 500 (zweryfikowane empirycznie: `curl localhost:3000/api/episodes` zwracalo 500 przed ta poprawka, 200 po niej). Byl to blad ukryty od poczatku uzycia docker-compose dla frontendu — poprzednie manualne testy w kolejnych etapach najwyrazniej sprawdzaly dzialanie backendu bezposrednio (`localhost:8000` z hosta, gdzie port jest wystawiony), nie przez rzeczywista sciezke przegladarki (`localhost:3000` -> proxy -> backend). `backend` to nazwa DNS uslugi w sieci compose, rozwiazywalna wylacznie wewnatrz kontenerow — bezpieczna do uzycia tutaj, bo `NEXT_PUBLIC_API_URL` mimo nazwy jest czytana wylacznie w konfiguracji serwera (`next.config.ts`), nigdy w kodzie wysylanym do przegladarki.

**Addendum (Etap 15 fix wave, 2026-09-09):** to zalozenie ("`NEXT_PUBLIC_API_URL` nigdy nie jest czytane po stronie przegladarki") okazalo sie nieaktualne, gdy Etap 15 dodal `frontend/lib/voiceWebSocket.ts`'s `voiceStreamUrl()` — kod uruchamiany W PRZEGLADARCE, ktory pierwotnie odczytywal wprost `NEXT_PUBLIC_API_URL`, zeby wyprowadzic adres WS (`http`->`ws`/`https`->`wss`). Pod `docker compose up` to bylo `http://backend:8000` — nazwa DNS rozwiazywalna wylacznie wewnatrz sieci compose, wiec `ws://backend:8000/...` nigdy nie laczyl sie z zadnej przegladarki na hoscie (streaming voice byl calkowicie niedzialajacy w lokalnym devie). Naprawiono dodajac osobna, faktycznie browser-facing zmienna `NEXT_PUBLIC_WS_URL` (ustawiona w `docker-compose.yml` na `ws://localhost:8000`), ktorej `voiceStreamUrl()` uzywa w pierwszej kolejnosci; gdy jest nieustawiona (produkcja — Vercel/Railway, gdzie backend ma prawdziwy publiczny URL `https://`), spada z powrotem na wyprowadzanie z `NEXT_PUBLIC_API_URL` jak dotychczas, bo tam ta derywacja jest poprawna. Wniosek: powyzsze uzasadnienie ("czytane wylacznie w `next.config.ts`") bylo prawdziwe wylacznie dla zwyklego HTTP-proxy przez `rewrites()` — nowy, oddzielny kanal (WebSocket, bez proxy) zlamal to zalozenie w sposob, ktorego Etap 14 nie mogl przewidziec.

---

## [Etap 15] Deploy produkcyjny: Railway (backend + Postgres jako `pgvector/pgvector:pg16` Docker image) + Vercel (frontend), bez zmian w `docker-compose.yml` dla lokalnego DX (2026-09-08)

**Decyzja:** `backend/railway.toml` wskazuje Railway na `backend/Dockerfile` (builder `dockerfile`) z healthcheckiem `/health`; Postgres wdrazany na Railway jako wlasny Docker image `pgvector/pgvector:pg16` (ten sam co lokalnie), nie przez domyslny plugin "PostgreSQL" Railwaya. `backend/Dockerfile` CMD zmieniony na `uvicorn ... --port ${PORT:-8000}` (bez `--reload`, z odczytem `$PORT` wymaganym przez Railway/Render), a `docker-compose.yml` dostal jawny `command:` z `--reload` dla lokalnego devu, zeby nie stracic hot-reloadu. Krok-po-kroku w `DEPLOYMENT.md`.

**Alternatywy odrzucone:** domyslny plugin Postgres na Railway (nie ma skompilowanego rozszerzenia `vector` — `CREATE EXTENSION vector` wywolywane przez `init_db` przy starcie backendu by sie wywalilo); Render zamiast Railway (rownowazna opcja, ale darmowy tier usypia po nieaktywnosci — gorsze pierwsze wrazenie dla rekrutera klikajacego link po przerwie); jeden VPS + docker-compose produkcyjny (prostsze pojedyncze miejsce, ale to staly koszt zamiast wykorzystania darmowych tierow Vercel/Railow, i wymaga wlasnego reverse proxy/TLS zamiast dostawania tego za darmo).

**Uzasadnienie:** Etap 13 juz przygotowal kod pod split-domain deploy (CORS policy, `X-Forwarded-For` w rate limiterze, brak hardkodowanego `localhost` w konfiguracji frontendu) — Etap 15 domyka to realnym planem wdrozenia, nie zmieniajac logiki aplikacji. Rozdzielenie CMD obrazu (produkcja, bez `--reload`, respektuje `$PORT`) od `command:` w compose (lokalny dev, z `--reload`) unika kompromisu miedzy "szybki hot-reload lokalnie" a "poprawne zachowanie na PaaS, ktory przydziela port dynamicznie". Wybor `pgvector/pgvector:pg16` jako wlasnego image zamiast plugin-Postgresa jest bezposrednia konsekwencja tego, ze `init_db` zaklada obecnosc rozszerzenia `vector` przy kazdym starcie — a nie kazdy zarzadzany Postgres je ma.

---

## [Etap 15] Pierwszy realny przebieg evala (NASA "Houston We Have a Podcast", 1 odcinek) ujawnia dwa realne bugi zamiast tylko liczb (2026-09-08)

**Decyzja:** Zaingestowano 1 realny odcinek (NASA "Houston We Have a Podcast" #435, "National Lab 15", ~24 min po ucięciu — Whisper API ma twardy limit 25MB/plik, oryginalny plik mial ~26-33MB w zależności od żądania z powodu dynamic ad insertion Megaphone; przycięto bajtowo do 24MB przed uploadem zamiast dodawać transkoding/ffmpeg do pipeline'u pod presją czasu — udokumentowane jako świadomy skrót, nie docelowe rozwiązanie). Napisano 8 pytań opartych na faktycznej treści transkryptu (`eval/golden_dataset/questions.json`) i uruchomiono `python -m eval.evaluate --modes semantic bm25 hybrid` (bez `hybrid+rerank` — brak `COHERE_API_KEY`).

Przy pierwszym uruchomieniu `GET /search?mode=semantic` zwracało 500 — `backend/app/search/searcher.py` uzywal `:emb::vector` w `text()`, a SQLAlchemy ma negative lookahead na `:` zaraz po nazwie bind-parametru (zeby nie kolidowac z operatorem rzutowania Postgresa `::`), wiec `:emb` nigdy nie bylo rozpoznawane jako parametr przy kompilacji — parametr byl cicho gubiony, zanim dotarl do sterownika. Naprawiono na `CAST(:emb AS vector)` i dodano test regresyjny (`test_semantic_search_query_actually_binds_emb_param`) kompilujacy zapytanie przeciwko prawdziwemu dialektowi postgres (bez zywej bazy) — zaden z 223 istniejacych testow tego nie zlapal, bo wszystkie mockuja sesje DB, wiec ten kod nigdy wczesniej nie wykonal sie na prawdziwym Postgresie.

**Alternatywy odrzucone:** dodanie ffmpeg/transkodingu do `ingest` przed wgraniem pod presja czasu ("chce miec live dzisiaj") — realny fix na docelowy etap, nie na szybko pod deadline; ukrywanie/ignorowanie niskiego wyniku BM25 zamiast go wytlumaczyc — wynik jest prawdziwy i pouczajacy, nie do schowania.

**Uzasadnienie:** Wyniki: semantic CR=0.812/GR=0.906/AR=0.938, bm25 CR=0.031/GR=0.031/AR=0.969, hybrid CR=0.812/GR=0.938/AR=0.938 (composite hybrid=0.896, najlepszy). BM25 dostaje **0 chunkow** dla 6 z 8 pytan — `bm25_search` buduje zapytanie przez `plainto_tsquery`, ktore ANDuje kazde slowo pytania; naturalne, 10-20-slowne pytania rzadko maja komplet swoich rdzeni w jednym ~60-100-slownym chunku. To dokladnie ten paraphrase-query failure mode, ktory hybrid/RRF (Etap 2) mial naprawiac — teraz potwierdzony liczbami, nie przewidywaniem z literatury. Answer Relevance zostaje wysokie nawet dla trybu bm25, bo `/agent/chat` zawsze wewnetrznie szuka w trybie `hybrid` niezaleznie od trybu, ktory eval-harness prosi z `/search` do oceny kontekstu — to naleza udokumentowac jako confound metodologii evala, nie jako "BM25 dziala dobrze mimo zerowego kontekstu".

---

## [Etap 15] Krytyczny bug streamingu: `/agent/chat/stream` nigdy nie wysylal zadnego tokena odpowiedzi (2026-09-08)

**Decyzja:** `backend/app/agent/streaming.py` odczytywal `event["data"]["chunk"].content` i sprawdzal `isinstance(token, str)`, zeby odfiltrowac puste/nietekstowe eventy. W realnej instalacji (`langchain-anthropic==1.7.1`, `langchain-core==1.6.2`) `chunk.content` to **lista blokow tresci** (`[{"type": "text", "text": "..."}]`, przeplatana z blokami `tool_use`/`input_json_delta` podczas tury wywolania narzedzia), nigdy plain string — wiec ten warunek odrzucal KAZDY chunk, zero eventow `token` nigdy nie bylo wysylanych. Dodano `_extract_text()`, ktore obsluguje oba ksztalty (str i liste blokow, filtrujac tylko `type == "text"`).

**Alternatywy odrzucone:** brak — to byl czysty bug, nie kompromis projektowy.

**Uzasadnienie:** Wykryte podczas realnego testowania na żywo przez uzytkownika ("no one answer from agent, all i get is fragments") — endpoint `/agent/chat/stream`, ktorego uzywa cala UI (tekst i glos), zwracal wylacznie zdarzenia `sources` + `done`, nigdy tresc odpowiedzi ani audio TTS (audio tez bylo zepsute, bo `SentenceChunker` nigdy nie dostawal tekstu do podzialu). Endpoint `/agent/chat` (nie-streamingowy, uzywany w evalu/etap 5) dzialal poprawnie, bo langchain zwraca tam `.content` na koncu jako scalony string, nie per-chunk podczas streamu — dlatego eval (ktory uzywa `/agent/chat`) nie wykazal tego problemu, mimo ze zostal uruchomiony na tym samym systemie tego samego dnia. To trzeci realny bug wykryty w Etapie 15 (po `searcher.py`'s `:emb::vector` i limicie rozmiaru pliku Whisper) o tym samym rdzeniu przyczyny: 225 mockowanych testow nigdy nie wykonalo tego kodu przeciwko prawdziwemu strumieniowi Anthropic — mock w `tests/agent/test_streaming.py::_make_token_event` ustawial `content` jako plain string. Naprawiono test helper tak, zeby mockowal realny ksztalt (lista blokow), i dodano `test_ignores_tool_use_blocks_interleaved_with_text`, ktory odtwarza dokladnie ten scenariusz (chunk z `tool_use`, chunk z `input_json_delta`, pusty chunk, potem tekst) — bez tej poprawki test failuje.

---

## [Etap 15] Krytyczny bug pamieci konwersacji: `thread_id` nic nie robil (2026-09-08)

**Decyzja:** `backend/app/agent/graph.py::build_graph()` tworzyl nowy `MemorySaver()` przy kazdym wywolaniu. Poniewaz zarowno `/agent/chat`, jak i `/agent/chat/stream` wywoluja `build_graph()` od nowa przy kazdym requescie HTTP (nowa sesja DB, nowy `sources_sink` na request), kazda wiadomosc dostawala **pusty** checkpoint store niezaleznie od `thread_id` — parametr `thread_id` byl calkowicie martwy, agent nie mial zadnej pamieci miedzy wiadomosciami, nawet w obrebie jednej sesji przegladarki. Przeniesiono `MemorySaver()` do modulowego singletona `_CHECKPOINTER`, wspoldzielonego przez wszystkie wywolania `build_graph()`.

**Alternatywy odrzucone:** trwaly checkpointer oparty o Postgres (np. `langgraph-checkpoint-postgres`) — poprawny kierunek na przyszlosc (przetrwalby restart backendu), ale to nowa zaleznosc i migracja, poza zakresem naprawy buga tego samego dnia; `MemorySaver` in-process pozostaje udokumentowanym, akceptowalnym ograniczeniem (pamiec ginie przy restarcie procesu) — teraz przynajmniej dziala w obrebie zycia procesu, co wczesniej nie mialo miejsca w ogole.

**Uzasadnienie:** Wykryte poprzez bezposrednie pytanie uzytkownika "czy tracimy historie rozmow w apce?" — zweryfikowane na zywo: pytanie uzupelniajace uzywajace zaimka ("he") z tym samym `thread_id` dostawalo "I need more context, who is 'he'?" zamiast poprawnej odpowiedzi. Po naprawie ten sam scenariusz dziala poprawnie i w jednym kroku (bez ponownego wywolania search_transcripts), bo odpowiedz byla juz w zapamietanej historii. Czwarty realny bug w Etapie 15 (po `searcher.py`, limicie Whisper, streaming content-shape) o tym samym wzorcu przyczynowym: `tests/test_agent_graph.py` mial 6 testow na `build_graph()`, zaden nie wywolywal go dwukrotnie z tym samym `thread_id`, wiec nikt nigdy nie sprawdzil, czy pamiec faktycznie przetrwa miedzy wywolaniami — dodano `test_thread_id_preserves_conversation_across_separate_build_graph_calls`, ktory to robi.

---

## [Etap 15] Trwala historia rozmow: `chat_threads`/`chat_messages` w Postgres, nie tylko checkpointer LangGraph (2026-09-08)

**Decyzja:** Dodano dwie tabele (`ChatThread`, `ChatMessage` w `backend/app/db/models.py`) i modul `backend/app/agent/history.py` (`ensure_thread`, `save_message`, `list_threads`, `get_thread`). `/agent/chat` i `/agent/chat/stream` zapisuja kazda wiadomosc uzytkownika i asystenta (z cytowaniami) do tych tabel; nowe endpointy `GET /agent/threads` (lista, sortowana po `updated_at`) i `GET /agent/threads/{id}` (pelna historia watku) obsluguja UI historii czatu w stylu Messengera. Zapis jest opakowany w `_persist_safely`/`_save_message_safely` — blad zapisu historii jest logowany, ale nigdy nie psuje realnej odpowiedzi czatu/glosu, ktorej uzytkownik akurat czeka.

**Alternatywy odrzucone:** poleganie wylacznie na `MemorySaver` (checkpointer LangGraph, Etap 15 wczesniejszy wpis) do "historii" — to pamiec robocza agenta (stan grafu potrzebny do kontynuacji rozmowy), nie user-facing log wiadomosci: nie przetrwa restartu procesu, nie ma tytulow watkow, nie da sie z niej zbudowac listy "otworz poprzednia rozmowe". Osobne tabele sa wlasciwym miejscem na trwaly, przegladalny zapis.

**Uzasadnienie:** Uzytkownik zapytal wprost "czy tracimy historie rozmow w apce, czy jest gdzies lista czatow do ponownego otwarcia" — odpowiedz przed tym wpisem brzmiala "nie ma nic". Tytul watku to pierwsze ~60 znakow pierwszej wiadomosci uzytkownika (`_make_title`), obcinane z "…".

**Bug znaleziony przy budowie tej funkcji:** Zapis wiadomosci asystenta w `stream_agent_response` byl pierwotnie umieszczony PO petli `await`-owania zadan TTS — ale awaria TTS jest tam celowo fatalna (patrz istniejacy test `test_tts_failure_yields_error_event_not_crash`: strumien ma sie urwac bledem, nie dojsc do `done`). Poniewaz `ELEVENLABS_API_KEY` byl przez cala sesje pusty, KAZDA odpowiedz strumieniowana konczyla sie awaria TTS, wiec zapis wiadomosci asystenta nigdy nie byl osiagany — tylko wiadomosc uzytkownika trafiala do bazy. Naprawiono przez przeniesienie liczenia `deduped` (cytowania) i zapisu wiadomosci asystenta na miejsce PRZED petla TTS — odpowiedz jest juz w tym momencie kompletna, TTS to efekt uboczny nizszego priorytetu niz zachowanie historii. Piaty realny bug w Etapie 15 wykryty przez rzeczywiste uzycie (live curl) zamiast tylko testow mockowanych — dodano `test_assistant_message_persisted_even_when_tts_fails`, ktory bez tej poprawki failuje.

---

## [Etap 15] Pierwszy realny deploy na Railway + Vercel — trzy pulapki platformowe (2026-09-09)

**Decyzja:** Backend + Postgres/pgvector wystawione na Railway (`voice-rag-agent-production-863d.up.railway.app`), frontend na Vercel (`voice-rag-agent-ten.vercel.app`). Zaktualizowano `DEPLOYMENT.md` o trzy realne pulapki napotkane przy pierwszym wdrozeniu — zaden z nich nie byl widoczny z poziomu kodu ani lokalnego docker-compose, wszystkie ujawnily sie dopiero na prawdziwej platformie.

**Pulapki:**
1. **Wolumen Railway z `lost+found`:** montowanie wolumenu bezposrednio na `/var/lib/postgresql/data` psuje `initdb` (katalog "nie jest pusty" — zawiera `lost+found` z formatowania systemu plikow). Fix: montowac na podkatalog (`/var/lib/postgresql/data/pgdata`) + `PGDATA` env var. Kontener wygladal jako "Active" w UI Railway mimo ze faktycznie byl w petli restartow.
2. **Rozjazd portu:** wygenerowanie publicznej domeny w Railway prosi o "target port", ale Railway moze niezaleznie przydzielic INNY losowy port jako faktyczny `$PORT` kontenera w runtime — trafilismy dokladnie na to (target 8000, kontener nasluchujacy na 8080), co dawalo 502 "Application failed to respond" mimo ze aplikacja wystartowala poprawnie. Fix: jawnie ustawic `PORT` jako zmienna srodowiskowa, zeby oba miejsca byly zgodne.
3. **GitHub App Railway bez dostepu do repo:** nowe konto Railway nie ma domyslnie zainstalowanej GitHub App na prywatnych/nowych repo, mimo ze publiczne repo daje sie polaczyc przez wklejenie URL. Objaw: "GitHub Repo not found" w Settings mimo poprawnie wybranego repo. Fix: github.com/settings/installations -> Configure -> dodac repo, potem rozlaczyc i polaczyc zrodlo w Railway (samo zapisanie Root Directory nie wymusza ponownego sprawdzenia dostepu).

**Uzasadnienie:** Zaden z etapow 1-14 (ani code review, ani 249 testow jednostkowych) nie mogl wykryc tych problemow — sa to wylacznie kwestie konfiguracji konkretnej platformy hostingowej, nie kodu aplikacji. Udokumentowane w `DEPLOYMENT.md` krok po kroku, zeby ponowny deploy (np. po utracie dostepu do tego konta Railway/Vercel) nie wymagal ponownego odkrywania tych samych trzech pulapek.

---

## [Etap 16] Przypisanie mowcy do chunku progiem >50% pokrycia czasowego, nie "wiekszosc slow" (2026-09-12)

**Decyzja:** `backend/app/ingest/diarizer.py::assign_speaker_labels` laczy kolejne slowa tego samego mowcy (z diaryzacji Deepgram) w segmenty, liczy pokrycie czasowe kazdego segmentu z przedzialem `[start_ts, end_ts]` chunku i przypisuje `speaker_label` tylko gdy jeden mowca pokrywa **>50%** dlugosci chunku — ponizej progu `speaker_label` zostaje `NULL`. Streszczenie i rozdzialy (`backend/app/ingest/summarizer.py::summarize_episode`) powstaja w jednym wywolaniu Claude Haiku wymuszonym przez tool-use (`tool_choice={"type": "tool", ...}`), nigdy przez parsowanie wolnego tekstu — transkrypt wysylany do modelu ma prefiks etykiety mowcy przy kazdej linii (`"Speaker N: ..."`), wiec `speaker_names` (mapowanie "Speaker N" -> prawdziwe imie) jest ugruntowane w etykietach, ktore model faktycznie widzial, a nie zgadywane.

**Alternatywy odrzucone:** przypisywanie mowcy po wiekszosci slow w chunku zamiast pokrycia czasowego — slowa krotkie ("tak", "no") mialyby ta sama wage co dlugie wypowiedzi, zniekszalcajac wynik przy przerywaniu sobie nawzajem; zgadywanie mowcy przy braku jednoznacznej wiekszosci — odrzucone explicite, bo cytowanie z bledna etykieta mowcy jest gorsze niz cytowanie bez etykiety; osobne wywolanie LLM per chapter/summary/speaker-mapping zamiast jednego wywolania ze wspolnym schematem — niepotrzebny koszt i ryzyko niespojnosci (np. rozne segmentacje czasowe miedzy chapters a summary), gdy jeden strukturalny tool-call daje spojny wynik za jednym zamachem.

**Uzasadnienie:** Ten sam rodzaj ryzyka co juz dwa razy uderzyl w tym projekcie (patrz `CLAUDE.md` "Known gotchas": ksztalt bloku tresci streamingu Anthropic, `:param::type` w SQLAlchemy) zostal swiadomie udokumentowany, nie ukryty: `diarize_audio` parsuje udokumentowany, ale **niezweryfikowany na zywo** ksztalt odpowiedzi Deepgram prerecorded API (`results.channels[0].alternatives[0].words[]` z polem `speaker` per slowo). Plan etapu 16 przewidywal osobne zadanie ("Task 10: live verification") wymagajace prawdziwego `DEEPGRAM_API_KEY`, realnego nagrania z wieloma mowcami i recznego odsluchu do potwierdzenia poprawnosci etykiet — ten krok nie zostal wykonany (brak `DEEPGRAM_API_KEY` w `.env`, brak takiego przebiegu w historii). Otwarte ryzyko: jesli realny ksztalt odpowiedzi Deepgrama odbiega od zalozonego, `diarize_audio` zawiedzie lub cicho zle sparsuje dane — zanim funkcja diaryzacji trafi na realny provisioning, wymagany jest jeden przebieg z prawdziwym kluczem na krotkim nagraniu wielomowcowym plus test regresyjny z przechwyconym realnym ksztaltem (analogicznie do `test_ignores_tool_use_blocks_interleaved_with_text` po buga streamingu Anthropic).

---

## [Etap 17] Jeden workflow GitHub Actions (`ci.yml`) z dwoma rownoleglymi jobami, bez sekretow (2026-09-12)

**Decyzja:** jeden workflow (`ci.yml`) z dwoma rownoleglymi jobami (`backend`, `frontend`), uruchamiany na kazdym push i pull request do `main` i raportujacy status obu jobow (widoczny przez status badge w README) — na tym etapie nic nie blokuje mergowania w razie niepowodzenia, bo branch protection na `main` nie zostal skonfigurowany. Bez sekretow API czy bazy danych — cala suita testów jest mockowana (300 testów backend, 48 testów frontend), nie wymaga real DB/API keys. Status badge w README.md linkuje do tego samego workflow file (`ci.yml`), zeby pozostal funkcjonalny niezaleznie od przyszlych rozszerzen.

**Alternatywy odrzucone:** osobne workflow files per job (np. `backend.yml`, `frontend.yml`) — niepotrzebna fragmentacja dla zaledwie dwoch jobow; uruchamianie testow w Dockerze wewnatrz CI zamiast bezposrednio na `ubuntu-latest` — wolniejsze (docker build time), niepotrzebne skoro testy nie potrzebuja realnej bazy (wszystko mockowane jak juz w CI); zmienne srodowiskowe CI (secrets) dla testow — projekt juz obsluguje brak kluczy (ANTHROPIC_API_KEY, OPENAI_API_KEY, itp sprawdzane per-request, nie przy starcie workera), wiec zero zmian w kodzie aplikacji dla CI.

**Uzasadnienie:** projekt osiagnal 348 testow (300 testów backend + 48 testów frontend) i zero automatyzacji ich uruchamiania — realna luka w projekcie pokazywanym rekruterom jako portfolio. Kazdego pusha na `main` czy PR do `main` powinno teraz automatycznie sprawdzic, czy suita przechodzi (status badge na README jest wizualnym dowodem). Pierwszy przebieg workflow-u (commit 425c4ef, run id 34696055262) przeszedl bez zadnych poprawek — caly workflow (oba joby, `backend` i `frontend`, uruchomione rownolegle na osobnych VM) zakonczyl sie z conclusion `success` w zweryfikowanym czasie ~67 sekund (utworzony 2026-09-12T13:18:16Z, zakonczony 2026-09-12T13:19:23Z). Brak surprises w zaleznosciach czy dlugosci instalacji — pytest i npm ci/test zadzialaly na ubuntu poprawnie juz za pierwszym razem.
