import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from eval.scorer import RAGTriadScorer, _parse_score, _format_contexts


def make_scorer(responses):
    calls = iter(responses)
    return RAGTriadScorer(llm_judge=lambda prompt: next(calls))


def test_parse_score_extracts_digit():
    assert _parse_score("5") == 1.0
    assert _parse_score("1") == 0.0
    assert _parse_score("3") == 0.5
    assert _parse_score("Some text 4 more") == 0.75


def test_parse_score_returns_half_on_garbage():
    assert _parse_score("no digits here") == 0.5


def test_format_contexts():
    out = _format_contexts(["hello", "world"])
    assert "[1] hello" in out
    assert "[2] world" in out


def test_score_context_relevance_returns_float():
    scorer = make_scorer(["4"])
    result = scorer.score_context_relevance("What is RAG?", ["RAG stands for..."])
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_context_relevance_empty_returns_zero():
    scorer = RAGTriadScorer(llm_judge=lambda p: "5")
    assert scorer.score_context_relevance("question", []) == 0.0


def test_score_groundedness_returns_float():
    scorer = make_scorer(["5"])
    result = scorer.score_groundedness("The answer is X", ["Context says X"])
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_groundedness_empty_answer_returns_zero():
    scorer = RAGTriadScorer(llm_judge=lambda p: "5")
    assert scorer.score_groundedness("", ["some context"]) == 0.0


def test_score_answer_relevance_returns_float():
    scorer = make_scorer(["3"])
    result = scorer.score_answer_relevance("What is AI?", "AI is artificial intelligence")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_answer_relevance_empty_returns_zero():
    scorer = RAGTriadScorer(llm_judge=lambda p: "5")
    assert scorer.score_answer_relevance("question", "") == 0.0


def test_score_all_returns_three_metrics():
    scorer = make_scorer(["4", "3", "5"])
    result = scorer.score_all("Q?", ["context"], "answer")
    assert set(result.keys()) == {"context_relevance", "groundedness", "answer_relevance"}
    assert all(isinstance(v, float) for v in result.values())


def test_score_all_high_scores():
    scorer = make_scorer(["5", "5", "5"])
    result = scorer.score_all("Q?", ["ctx"], "ans")
    assert result["context_relevance"] == 1.0
    assert result["groundedness"] == 1.0
    assert result["answer_relevance"] == 1.0


def test_score_all_low_scores():
    scorer = make_scorer(["1", "1", "1"])
    result = scorer.score_all("Q?", ["ctx"], "ans")
    assert result["context_relevance"] == 0.0
    assert result["groundedness"] == 0.0
    assert result["answer_relevance"] == 0.0
