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
