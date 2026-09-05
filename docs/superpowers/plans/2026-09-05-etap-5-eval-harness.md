# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG triad evaluation harness that measures retrieval quality across all four search modes using an LLM-as-judge, producing a reproducible score table for the portfolio README.

**Architecture:** Separate `eval/` package (no backend import dependency); calls the running backend via HTTP. RAGTriadScorer wraps an LLM judge callable so the judge is swappable. EvalRunner orchestrates per-question HTTP calls and error capture. evaluate.py is a CLI entry point.

**Tech Stack:** Python 3.12, httpx, anthropic (Messages API), pytest, json, statistics

---

## File Structure

| File | Responsibility |
|------|---------------|
| `eval/__init__.py` | Package marker |
| `eval/scorer.py` | `RAGTriadScorer` class: context_relevance, groundedness, answer_relevance; `make_anthropic_judge()` |
| `eval/runner.py` | `EvalRunner`: calls GET /search + POST /agent/chat per question/mode, captures errors |
| `eval/evaluate.py` | CLI entry point; aggregates results; writes JSON + Markdown reports |
| `eval/golden_dataset/questions.json` | 15 synthetic Q&A pairs (5 simple, 5 comparison, 5 multi-hop) |
| `backend/tests/test_eval_scorer.py` | 12 unit tests for scorer (mocked judge) |
| `backend/tests/test_eval_runner.py` | 6 unit tests for runner (mocked httpx) |
| `backend/tests/test_eval_evaluate.py` | 8 unit tests for aggregate + render + load |

---

### Task 1: Golden dataset

**Files:**
- Create: `eval/golden_dataset/questions.json`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_eval_evaluate.py
def test_load_questions_returns_list():
    questions = load_questions()
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert "question" in questions[0]
    assert "id" in questions[0]

def test_questions_have_difficulty():
    questions = load_questions()
    for q in questions:
        assert q["difficulty"] in ("simple", "multi_hop", "comparison")
```

- [ ] **Step 2: Run test — expect FAIL** (file doesn't exist)

```bash
cd backend && python -m pytest tests/test_eval_evaluate.py::test_load_questions_returns_list -v
```

- [ ] **Step 3: Create `eval/golden_dataset/questions.json`** — 15 entries with `{id, question, expected_topics, difficulty}`. Include 5 simple, 5 comparison, 5 multi-hop questions.

- [ ] **Step 4: Create `eval/evaluate.py`** with `load_questions(limit=None)` that reads the JSON file.

- [ ] **Step 5: Run tests — expect PASS**

```bash
cd backend && python -m pytest tests/test_eval_evaluate.py -v
```

- [ ] **Step 6: Commit**

```bash
git add eval/golden_dataset/questions.json eval/evaluate.py
git commit -m "test(eval): add golden dataset and load_questions"
```

---

### Task 2: RAG Triad Scorer

**Files:**
- Create: `eval/scorer.py`
- Create: `backend/tests/test_eval_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_eval_scorer.py
def test_score_context_relevance_returns_float():
    scorer = RAGTriadScorer(llm_judge=lambda p: "4")
    result = scorer.score_context_relevance("What is RAG?", ["RAG stands for..."])
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_score_context_relevance_empty_returns_zero():
    scorer = RAGTriadScorer(llm_judge=lambda p: "5")
    assert scorer.score_context_relevance("question", []) == 0.0

def test_score_all_returns_three_metrics():
    scorer = RAGTriadScorer(llm_judge=lambda p: "4")
    result = scorer.score_all("Q?", ["context"], "answer")
    assert set(result.keys()) == {"context_relevance", "groundedness", "answer_relevance"}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `eval/scorer.py`**

```python
class RAGTriadScorer:
    def __init__(self, llm_judge):
        self._judge = llm_judge  # callable(prompt: str) -> str with "1"-"5"

    def score_context_relevance(self, question, contexts) -> float:
        if not contexts: return 0.0
        # prompt asks LLM to rate 1-5; _parse_score normalises to (val-1)/4.0
        return _parse_score(self._judge(_CONTEXT_RELEVANCE_PROMPT.format(...)))

    def score_groundedness(self, answer, contexts) -> float: ...
    def score_answer_relevance(self, question, answer) -> float: ...
    def score_all(self, question, contexts, answer) -> dict: ...

def _parse_score(text: str) -> float:
    for ch in text:
        if ch.isdigit() and 1 <= int(ch) <= 5:
            return (int(ch) - 1) / 4.0
    return 0.5

def make_anthropic_judge(api_key, model="claude-haiku-4-5-20251001"):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    def judge(prompt):
        msg = client.messages.create(model=model, max_tokens=16,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    return judge
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add eval/scorer.py backend/tests/test_eval_scorer.py
git commit -m "feat(eval): add RAGTriadScorer with LLM-as-judge"
```

---

### Task 3: EvalRunner

**Files:**
- Create: `eval/runner.py`
- Create: `backend/tests/test_eval_runner.py`

- [ ] **Step 1: Write failing tests**

```python
def test_run_question_calls_search_and_agent():
    runner = EvalRunner(scorer=make_scorer())
    with patch("eval.runner.httpx.get", return_value=make_search_response()), \
         patch("eval.runner.httpx.post", return_value=make_chat_response()):
        result = runner.run_question("What is AI?", "semantic")
    assert result["answer"] == "The answer is X"
    assert result["mode"] == "semantic"

def test_run_dataset_captures_errors():
    runner = EvalRunner(scorer=make_scorer())
    with patch("eval.runner.httpx.get", side_effect=Exception("refused")):
        results = runner.run_dataset([{"id":"q1","question":"Q?"}], ["semantic"])
    assert "error" in results[0]
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `eval/runner.py`**

```python
class EvalRunner:
    def __init__(self, scorer, base_url="http://localhost:8000", search_limit=5, timeout=30.0):
        ...

    def run_question(self, question, mode) -> dict:
        chunks = httpx.get(f"{self._base_url}/search", params={...}).json()
        answer = httpx.post(f"{self._base_url}/agent/chat", json={...}).json()["reply"]
        scores = self._scorer.score_all(question, [c["text"] for c in chunks], answer)
        return {"question": question, "mode": mode, "num_chunks": len(chunks),
                "answer": answer, "scores": scores}

    def run_dataset(self, questions, modes) -> list[dict]:
        results = []
        for q in questions:
            for mode in modes:
                try: results.append({**self.run_question(q["question"], mode), "id": q["id"]})
                except Exception as e: results.append({"id": q["id"], "error": str(e), ...})
        return results
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add eval/runner.py backend/tests/test_eval_runner.py
git commit -m "feat(eval): add EvalRunner with per-question error capture"
```

---

### Task 4: CLI + report generation

**Files:**
- Complete: `eval/evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
def test_aggregate_produces_per_mode_means():
    results = make_results(modes=["semantic", "hybrid"])
    summary = aggregate(results)
    assert "semantic" in summary
    assert summary["semantic"]["context_relevance"] == 0.75

def test_aggregate_includes_composite():
    results = make_results(modes=["semantic"])
    summary = aggregate(results)
    expected = round((0.75 + 0.5 + 0.8) / 3, 4)
    assert summary["semantic"]["composite"] == expected

def test_render_markdown_contains_table():
    md = render_markdown(summary, results, "2026-09-05")
    assert "| Mode |" in md
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `aggregate()`, `render_markdown()`, and `main()` in `eval/evaluate.py`**

```python
def aggregate(results) -> dict:
    # per_mode[mode][metric] = [scores]; return means + composite

def render_markdown(summary, results, timestamp) -> str:
    # Returns markdown table: modes × metrics + per-question rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES)
    parser.add_argument("--output", default="docs/eval_report")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    judge = make_anthropic_judge(api_key)
    scorer = RAGTriadScorer(judge)
    runner = EvalRunner(scorer=scorer, base_url=args.base_url)
    questions = load_questions(args.limit)
    results = runner.run_dataset(questions, args.modes)
    summary = aggregate(results)
    # write JSON + MD to args.output.{json,md}
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add eval/evaluate.py backend/tests/test_eval_evaluate.py
git commit -m "feat(eval): add CLI runner with aggregate + markdown report"
```

---

### Task 5: Full suite verification

- [ ] **Step 1: Run full backend suite**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: **115 passed** (89 prior + 26 new eval tests).

- [ ] **Step 2: Smoke test CLI with `--limit 1` (requires running backend)**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m eval.evaluate --modes semantic --limit 1 --output /tmp/eval_test
cat /tmp/eval_test.md
```

- [ ] **Step 3: Squash-commit to main**

```bash
git reset --soft $(git merge-base HEAD main)
git commit -m "feat(etap-5): add RAG eval harness with LLM-as-judge and golden dataset"
```
