"""MCP tool implementations — pure functions that call the backend REST API."""

from __future__ import annotations

import httpx


def search_knowledge_base(
    query: str,
    base_url: str = "http://localhost:8000",
    limit: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    """Search podcast/lecture transcripts using hybrid retrieval.

    Returns a list of chunks: {chunk_id, episode_id, start_ts, end_ts, text, similarity}.
    """
    resp = httpx.get(
        f"{base_url}/search",
        params={"q": query, "mode": mode, "limit": limit},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_episode_summary(
    episode_id: str,
    base_url: str = "http://localhost:8000",
    chunk_limit: int = 10,
) -> dict:
    """Return metadata and the first N transcript chunks for an episode.

    Returns {episode, chunks: [{start_ts, end_ts, text}, ...]}.
    """
    ep_resp = httpx.get(f"{base_url}/episodes/{episode_id}", timeout=10.0)
    ep_resp.raise_for_status()
    episode = ep_resp.json()

    chunks_resp = httpx.get(
        f"{base_url}/episodes/{episode_id}/chunks",
        params={"limit": chunk_limit},
        timeout=10.0,
    )
    chunks_resp.raise_for_status()
    chunks = chunks_resp.json()

    return {"episode": episode, "chunks": chunks}


def find_mentions(
    topic: str,
    base_url: str = "http://localhost:8000",
    limit: int = 10,
) -> list[dict]:
    """Find all places in the library where a topic is mentioned.

    Returns a list of chunks sorted by relevance, each with episode_id and timestamp.
    """
    resp = httpx.get(
        f"{base_url}/search",
        params={"q": topic, "mode": "hybrid", "limit": limit},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


def format_chunks_for_mcp(chunks: list[dict]) -> str:
    """Format search results into a readable string for MCP tool output."""
    if not chunks:
        return "No results found."
    lines = []
    for i, c in enumerate(chunks, 1):
        start = c.get("start_ts", 0)
        end = c.get("end_ts", 0)
        ep = c.get("episode_id", "unknown")
        text = c.get("text", "")
        lines.append(f"[{i}] Episode {ep} @ {start:.1f}s–{end:.1f}s\n{text}")
    return "\n\n".join(lines)
