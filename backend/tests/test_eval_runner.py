import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from unittest.mock import MagicMock, patch
from eval.scorer import RAGTriadScorer
from eval.runner import EvalRunner


def make_scorer():
    scorer = MagicMock(spec=RAGTriadScorer)
    scorer.score_all.return_value = {
        "context_relevance": 0.75,
        "groundedness": 0.5,
        "answer_relevance": 0.8,
    }
    return scorer


def make_search_response(texts=None):
    texts = texts or ["chunk text 1", "chunk text 2"]
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = [
        {"chunk_id": f"c{i}", "text": t, "episode_id": "ep1", "start_ts": 0.0, "end_ts": 10.0}
        for i, t in enumerate(texts)
    ]
    return mock


def make_chat_response(reply="The answer is X"):
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"reply": reply, "steps": 2}
    return mock


def test_run_question_calls_search_and_agent():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer, base_url="http://localhost:8000")
    with patch("eval.runner.httpx.get", return_value=make_search_response()) as mock_get, \
         patch("eval.runner.httpx.post", return_value=make_chat_response()) as mock_post:
        result = runner.run_question("What is AI?", "semantic")

    mock_get.assert_called_once()
    mock_post.assert_called_once()
    assert result["question"] == "What is AI?"
    assert result["mode"] == "semantic"
    assert result["answer"] == "The answer is X"


def test_run_question_extracts_chunk_texts():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer)
    with patch("eval.runner.httpx.get", return_value=make_search_response(["ctx1", "ctx2"])), \
         patch("eval.runner.httpx.post", return_value=make_chat_response()):
        result = runner.run_question("Q?", "hybrid")

    scorer.score_all.assert_called_once_with("Q?", ["ctx1", "ctx2"], "The answer is X")


def test_run_question_returns_scores():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer)
    with patch("eval.runner.httpx.get", return_value=make_search_response()), \
         patch("eval.runner.httpx.post", return_value=make_chat_response()):
        result = runner.run_question("Q?", "bm25")

    assert result["scores"]["context_relevance"] == 0.75
    assert result["scores"]["groundedness"] == 0.5
    assert result["scores"]["answer_relevance"] == 0.8


def test_run_dataset_iterates_modes():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer)
    questions = [{"id": "q1", "question": "Q?", "difficulty": "simple"}]
    modes = ["semantic", "bm25"]
    with patch("eval.runner.httpx.get", return_value=make_search_response()), \
         patch("eval.runner.httpx.post", return_value=make_chat_response()):
        results = runner.run_dataset(questions, modes)

    assert len(results) == 2
    assert results[0]["mode"] == "semantic"
    assert results[1]["mode"] == "bm25"


def test_run_dataset_captures_errors():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer)
    questions = [{"id": "q1", "question": "Q?"}]
    with patch("eval.runner.httpx.get", side_effect=Exception("connection refused")):
        results = runner.run_dataset(questions, ["semantic"])

    assert len(results) == 1
    assert "error" in results[0]
    assert "connection refused" in results[0]["error"]
    assert results[0]["scores"]["context_relevance"] == 0.0


def test_run_question_records_num_chunks():
    scorer = make_scorer()
    runner = EvalRunner(scorer=scorer)
    with patch("eval.runner.httpx.get", return_value=make_search_response(["a", "b", "c"])), \
         patch("eval.runner.httpx.post", return_value=make_chat_response()):
        result = runner.run_question("Q?", "hybrid+rerank")

    assert result["num_chunks"] == 3
