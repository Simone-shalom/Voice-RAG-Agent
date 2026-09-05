# Etap 5: Eval Harness

## What was added

**`eval/scorer.py`** — `RAGTriadScorer`: three metrics, each 0-1 via an LLM judge callable:
- `score_context_relevance(question, contexts)` — do chunks contain relevant info?
- `score_groundedness(answer, contexts)` — is the answer supported by chunks?
- `score_answer_relevance(question, answer)` — does the answer address the question?

Judge prompts ask for 1-5 integer; `_parse_score` normalises to `(val-1)/4.0`. `make_anthropic_judge(api_key)` wires Anthropic Messages API as the backend.

**`eval/runner.py`** — `EvalRunner`: calls `GET /search?q=...&mode=...` then `POST /agent/chat`, extracts chunk texts, calls scorer. Errors per question/mode are captured (not raised), so one timeout doesn't abort the whole run.

**`eval/evaluate.py`** — CLI entry point:
```bash
python -m eval.evaluate \
  --base-url http://localhost:8000 \
  --modes semantic bm25 hybrid hybrid+rerank \
  --output docs/eval_report \
  --limit 5    # optional, for quick smoke test
```
Requires: `ANTHROPIC_API_KEY` in env. Outputs `docs/eval_report.json` + `docs/eval_report.md`.

**`eval/golden_dataset/questions.json`** — 15 synthetic questions:
- 5 `simple`: single-hop factual
- 5 `comparison`: cross-episode comparison
- 5 `multi_hop`: require agent tool-calling (multi-step retrieval)

Difficulty spread is intentional — simple questions don't expose differences between retrieval modes.

## Test counts (26 new)
- `test_eval_scorer.py` — 12 tests: parse_score, format_contexts, all three metrics, empty inputs, score_all
- `test_eval_runner.py` — 6 tests: search+agent calls, chunk text extraction, error capture, num_chunks
- `test_eval_evaluate.py` — 8 tests: aggregate, render_markdown, load_questions, limit

Total backend: **115 passing**.

## Key decisions
See `DECISIONS.md`: LLM-as-judge rationale, 1-5 integer score design.
