#!/usr/bin/env python3
"""
Eval harness: runs RAG triad evaluation across retrieval modes.

Usage:
    python evaluate.py [--base-url URL] [--modes MODE...] [--output PATH] [--limit N]

Example:
    python evaluate.py --modes semantic hybrid hybrid+rerank --output docs/eval_report
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from .scorer import RAGTriadScorer, make_anthropic_judge
from .runner import EvalRunner

DATASET_PATH = Path(__file__).parent / "golden_dataset" / "questions.json"
ALL_MODES = ["semantic", "bm25", "hybrid", "hybrid+rerank"]
METRICS = ["context_relevance", "groundedness", "answer_relevance"]


def load_questions(limit: int | None = None) -> list[dict]:
    with open(DATASET_PATH) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    return questions


def aggregate(results: list[dict]) -> dict:
    per_mode: dict[str, dict[str, list[float]]] = {}
    for r in results:
        mode = r["mode"]
        if mode not in per_mode:
            per_mode[mode] = {m: [] for m in METRICS}
        for metric in METRICS:
            per_mode[mode][metric].append(r["scores"][metric])

    summary = {}
    for mode, metric_scores in per_mode.items():
        summary[mode] = {
            metric: round(statistics.mean(scores), 4) if scores else 0.0
            for metric, scores in metric_scores.items()
        }
        # composite: mean of all three metrics
        vals = [summary[mode][m] for m in METRICS]
        summary[mode]["composite"] = round(statistics.mean(vals), 4)

    return summary


def render_markdown(summary: dict, results: list[dict], timestamp: str) -> str:
    lines = [
        "# RAG Eval Report",
        f"\n_Generated: {timestamp}_\n",
        "## Summary — mean scores per retrieval mode\n",
        "| Mode | Context Relevance | Groundedness | Answer Relevance | Composite |",
        "|------|:-----------------:|:------------:|:----------------:|:---------:|",
    ]
    for mode in ALL_MODES:
        if mode not in summary:
            continue
        s = summary[mode]
        lines.append(
            f"| {mode} | {s['context_relevance']:.3f} | {s['groundedness']:.3f} "
            f"| {s['answer_relevance']:.3f} | {s['composite']:.3f} |"
        )

    lines += [
        "\n## Per-question results\n",
        "| ID | Difficulty | Mode | CR | GR | AR |",
        "|----|-----------|------|----|----|----|",
    ]
    for r in results:
        s = r["scores"]
        lines.append(
            f"| {r.get('id','?')} | {r.get('difficulty','?')} | {r['mode']} "
            f"| {s['context_relevance']:.2f} | {s['groundedness']:.2f} | {s['answer_relevance']:.2f} |"
        )

    return "\n".join(lines) + "\n"


def run_eval(
    base_url: str,
    modes: list[str],
    output: str,
    limit: int | None,
    api_key: str,
) -> None:
    judge = make_anthropic_judge(api_key)
    scorer = RAGTriadScorer(llm_judge=judge)
    runner = EvalRunner(scorer=scorer, base_url=base_url)

    questions = load_questions(limit)
    print(f"Running {len(questions)} questions × {len(modes)} modes = {len(questions)*len(modes)} evals…")

    results = runner.run_dataset(questions, modes)

    summary = aggregate(results)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"timestamp": timestamp, "summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"JSON report written to {json_path}")

    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_markdown(summary, results, timestamp), encoding="utf-8")
    print(f"Markdown report written to {md_path}")

    print("\n=== Summary ===")
    for mode, scores in summary.items():
        print(f"{mode:20s}  CR={scores['context_relevance']:.3f}  GR={scores['groundedness']:.3f}  AR={scores['answer_relevance']:.3f}  composite={scores['composite']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval harness")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--output", default="docs/eval_report")
    parser.add_argument("--limit", type=int, default=None, help="limit number of questions (for quick smoke tests)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    run_eval(
        base_url=args.base_url,
        modes=args.modes,
        output=args.output,
        limit=args.limit,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
