import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

from eval.evaluate import aggregate, render_markdown, load_questions


def make_results(modes=("semantic", "hybrid"), num_questions=2):
    results = []
    for i in range(num_questions):
        for mode in modes:
            results.append({
                "id": f"q{i+1:02d}",
                "question": f"Question {i+1}?",
                "mode": mode,
                "difficulty": "simple",
                "answer": "some answer",
                "num_chunks": 5,
                "scores": {
                    "context_relevance": 0.75,
                    "groundedness": 0.5,
                    "answer_relevance": 0.8,
                },
            })
    return results


def test_aggregate_produces_per_mode_means():
    results = make_results(modes=["semantic", "hybrid"])
    summary = aggregate(results)
    assert "semantic" in summary
    assert "hybrid" in summary
    assert summary["semantic"]["context_relevance"] == 0.75
    assert summary["semantic"]["groundedness"] == 0.5


def test_aggregate_includes_composite():
    results = make_results(modes=["semantic"])
    summary = aggregate(results)
    expected_composite = round((0.75 + 0.5 + 0.8) / 3, 4)
    assert summary["semantic"]["composite"] == expected_composite


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}


def test_render_markdown_contains_table():
    results = make_results(modes=["semantic"])
    summary = aggregate(results)
    md = render_markdown(summary, results, "2026-09-05T00:00:00Z")
    assert "| Mode |" in md
    assert "semantic" in md
    assert "2026-09-05T00:00:00Z" in md


def test_render_markdown_per_question_rows():
    results = make_results(modes=["semantic"], num_questions=3)
    summary = aggregate(results)
    md = render_markdown(summary, results, "ts")
    assert "q01" in md
    assert "q02" in md
    assert "q03" in md


def test_load_questions_returns_list():
    questions = load_questions()
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert "question" in questions[0]
    assert "id" in questions[0]


def test_load_questions_limit():
    questions = load_questions(limit=3)
    assert len(questions) == 3


def test_questions_have_difficulty():
    questions = load_questions()
    for q in questions:
        assert q["difficulty"] in ("simple", "multi_hop", "comparison")
