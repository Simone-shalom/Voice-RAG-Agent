#!/usr/bin/env python3
"""
Compare all four search modes on a single query.

Usage:
    cd <project-root>
    python -m eval.search_comparison "your search query" [--limit 5]

Requires:
    DATABASE_URL and OPENAI_API_KEY set in environment or .env file
    COHERE_API_KEY set for hybrid+rerank mode (optional; that mode is skipped otherwise)
"""
import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.search.bm25 import bm25_search
from app.search.searcher import hybrid_rerank_search, hybrid_search, semantic_search


def get_session():
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    engine = create_engine(url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session()


def fmt_result(i: int, chunk: dict) -> str:
    ts = f"{chunk['start_ts']:.1f}s–{chunk['end_ts']:.1f}s"
    score = f"{chunk.get('similarity', 0.0):.4f}"
    preview = chunk["text"][:80].replace("\n", " ")
    return f"{i+1}. [{score}] {ts} | {preview}"


def run_mode(name: str, fn, query: str, limit: int, db) -> list[str]:
    try:
        results = fn(query=query, limit=limit, db=db)
        return [fmt_result(i, r) for i, r in enumerate(results)]
    except Exception as e:
        return [f"ERROR: {e}"]


def main():
    parser = argparse.ArgumentParser(description="Compare search modes")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    db = get_session()
    query = args.query
    limit = args.limit

    modes = {
        "semantic":      lambda q, l, d: semantic_search(q, l, d),
        "bm25":          lambda q, l, d: bm25_search(q, l, d),
        "hybrid":        lambda q, l, d: hybrid_search(q, l, d),
        "hybrid+rerank": lambda q, l, d: hybrid_rerank_search(q, l, d),
    }

    lines = [f"# Search comparison: '{query}' (limit={limit})\n"]
    for mode_name, fn in modes.items():
        lines.append(f"## {mode_name}")
        lines.extend(run_mode(mode_name, fn, query, limit, db))
        lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "search_comparison_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {out_path}")

    db.close()


if __name__ == "__main__":
    main()
