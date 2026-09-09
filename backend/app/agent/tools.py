import os

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..search.bm25 import bm25_search
from ..search.searcher import hybrid_rerank_search, hybrid_search, semantic_search

MAX_CONTEXT_CHARS = 2000


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No results found."
    lines = []
    for c in chunks:
        ts = f"{c['start_ts']:.1f}s–{c['end_ts']:.1f}s"
        lines.append(f"[{c['episode_id']} @ {ts}] {c['text']}")
    return "\n".join(lines)


def make_tools(db: Session, mode: str = "hybrid", sources_sink: list[dict] | None = None) -> list:
    """
    Build the agent's tool list with db session captured in closure.

    mode: which search function search_transcripts/compare_across_episodes use
      ("semantic" | "bm25" | "hybrid" | "hybrid+rerank"); falls back to hybrid_search
      for an unrecognised value.
    sources_sink: if given, every chunk retrieved by search_transcripts,
      get_context_around_timestamp, or compare_across_episodes is appended to it —
      lets the caller (streaming.py) collect what was actually cited in the answer.

    Returns 4 LangChain @tool instances.
    """
    search_funcs = {
        "semantic": semantic_search,
        "bm25": bm25_search,
        "hybrid": hybrid_search,
        "hybrid+rerank": hybrid_rerank_search,
    }
    if mode == "hybrid+rerank" and not os.environ.get("COHERE_API_KEY"):
        mode = "hybrid"  # avoid a hard tool failure when reranking isn't configured
    search_fn = search_funcs.get(mode, hybrid_search)
    if sources_sink is None:
        sources_sink = []

    @tool
    def search_transcripts(query: str, limit: int = 5) -> str:
        """Search across all podcast transcripts using hybrid search (semantic + BM25 + RRF).
        Returns the most relevant chunks with timestamps."""
        limit = min(max(limit, 1), 20)
        results = search_fn(query=query, limit=limit, db=db)
        sources_sink.extend(results)
        return _format_chunks(results)

    @tool
    def get_context_around_timestamp(episode_id: str, timestamp: float, window_seconds: float = 30.0) -> str:
        """Retrieve transcript chunks from a specific episode near a given timestamp.
        Use this to get more context around a citation you found with search_transcripts."""
        rows = db.execute(
            text(
                """
                SELECT id, episode_id, start_ts, end_ts, text, speaker_label
                FROM chunks
                WHERE episode_id = :ep_id
                  AND start_ts >= :start AND end_ts <= :end
                ORDER BY start_ts
                LIMIT 10
                """
            ),
            {
                "ep_id": episode_id,
                "start": max(0.0, timestamp - window_seconds / 2),
                "end": timestamp + window_seconds / 2,
            },
        ).fetchall()
        if not rows:
            return f"No chunks found in episode {episode_id} near {timestamp:.1f}s."
        for r in rows:
            sources_sink.append({
                "episode_id": str(r.episode_id),
                "start_ts": r.start_ts,
                "end_ts": r.end_ts,
                "text": r.text,
                "speaker_label": r.speaker_label,
            })
        return "\n".join(f"[{r.start_ts:.1f}s–{r.end_ts:.1f}s] {r.text}" for r in rows)

    @tool
    def compare_across_episodes(query: str, episode_ids: str, limit_per_episode: int = 3) -> str:
        """Search for a topic within specific episodes and return results grouped by episode.
        episode_ids: comma-separated episode UUIDs.
        Use this when the user asks to compare what different episodes say about the same topic."""
        limit_per_episode = min(max(limit_per_episode, 1), 20)
        ids = [e.strip() for e in episode_ids.split(",") if e.strip()]
        all_results = search_fn(query=query, limit=limit_per_episode * max(len(ids), 1), db=db)
        by_episode: dict[str, list[dict]] = {}
        for chunk in all_results:
            ep = chunk["episode_id"]
            if ep in ids:
                by_episode.setdefault(ep, []).append(chunk)
        if not by_episode:
            return f"No results for '{query}' in the specified episodes."
        parts = []
        for ep_id in ids:
            chunks = by_episode.get(ep_id, [])[:limit_per_episode]
            sources_sink.extend(chunks)
            parts.append(f"=== Episode {ep_id} ===")
            parts.append(_format_chunks(chunks))
        return "\n".join(parts)

    @tool
    def summarise_segment(text: str) -> str:
        """Format a transcript segment for the assistant to summarise.
        Pass the raw transcript text; returns it formatted for analysis."""
        truncated = text[:MAX_CONTEXT_CHARS]
        return f"[SEGMENT TO SUMMARISE]\n{truncated}"

    return [search_transcripts, get_context_around_timestamp, compare_across_episodes, summarise_segment]
