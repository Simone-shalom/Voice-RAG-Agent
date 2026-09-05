from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..search.searcher import hybrid_search

MAX_CONTEXT_CHARS = 2000


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No results found."
    lines = []
    for c in chunks:
        ts = f"{c['start_ts']:.1f}s–{c['end_ts']:.1f}s"
        lines.append(f"[{c['episode_id']} @ {ts}] {c['text']}")
    return "\n".join(lines)


def make_tools(db: Session) -> list:
    """
    Build the agent's tool list with db session captured in closure.
    Returns 4 LangChain @tool instances.
    """

    @tool
    def search_transcripts(query: str, limit: int = 5) -> str:
        """Search across all podcast transcripts using hybrid search (semantic + BM25 + RRF).
        Returns the most relevant chunks with timestamps."""
        results = hybrid_search(query=query, limit=limit, db=db)
        return _format_chunks(results)

    @tool
    def get_context_around_timestamp(episode_id: str, timestamp: float, window_seconds: float = 30.0) -> str:
        """Retrieve transcript chunks from a specific episode near a given timestamp.
        Use this to get more context around a citation you found with search_transcripts."""
        rows = db.execute(
            text(
                """
                SELECT id, episode_id, start_ts, end_ts, text
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
        return "\n".join(f"[{r.start_ts:.1f}s–{r.end_ts:.1f}s] {r.text}" for r in rows)

    @tool
    def compare_across_episodes(query: str, episode_ids: str, limit_per_episode: int = 3) -> str:
        """Search for a topic within specific episodes and return results grouped by episode.
        episode_ids: comma-separated episode UUIDs.
        Use this when the user asks to compare what different episodes say about the same topic."""
        ids = [e.strip() for e in episode_ids.split(",") if e.strip()]
        all_results = hybrid_search(query=query, limit=limit_per_episode * max(len(ids), 1), db=db)
        by_episode: dict[str, list[dict]] = {}
        for chunk in all_results:
            ep = chunk["episode_id"]
            if ep in ids:
                by_episode.setdefault(ep, []).append(chunk)
        if not by_episode:
            return f"No results for '{query}' in the specified episodes."
        parts = []
        for ep_id in ids:
            chunks = by_episode.get(ep_id, [])
            parts.append(f"=== Episode {ep_id} ===")
            parts.append(_format_chunks(chunks[:limit_per_episode]))
        return "\n".join(parts)

    @tool
    def summarise_segment(text: str) -> str:
        """Format a transcript segment for the assistant to summarise.
        Pass the raw transcript text; returns it formatted for analysis."""
        truncated = text[:MAX_CONTEXT_CHARS]
        return f"[SEGMENT TO SUMMARISE]\n{truncated}"

    return [search_transcripts, get_context_around_timestamp, compare_across_episodes, summarise_segment]
