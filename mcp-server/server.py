#!/usr/bin/env python3
"""Voice Knowledge Agent MCP Server.

Exposes the podcast/lecture knowledge base as MCP tools for Claude Desktop / Claude Code.

Usage:
    python mcp-server/server.py

Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "voice-knowledge": {
          "command": "python",
          "args": ["<absolute-path-to-repo>/mcp-server/server.py"],
          "env": {}
        }
      }
    }
"""

import os
import sys

# Allow running as `python mcp-server/server.py` from repo root
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from tools import (
    find_mentions,
    format_chunks_for_mcp,
    get_episode_summary,
    search_knowledge_base,
)

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

mcp = FastMCP("Voice Knowledge Agent")


@mcp.tool()
def search_knowledge_base_tool(query: str, limit: int = 5) -> str:
    """Search across podcast and lecture transcripts using hybrid semantic+BM25 retrieval.

    Args:
        query: The search question or topic (plain text).
        limit: Maximum number of result chunks to return (default 5, max 20).

    Returns:
        Formatted list of transcript chunks with episode ID and timestamp.
    """
    limit = min(limit, 20)
    chunks = search_knowledge_base(query, base_url=BASE_URL, limit=limit)
    return format_chunks_for_mcp(chunks)


@mcp.tool()
def get_episode_summary_tool(episode_id: str) -> str:
    """Get metadata and the opening transcript of a specific episode.

    Args:
        episode_id: UUID of the episode (obtain from search results).

    Returns:
        Episode filename, source URL, chunk count, and first 10 transcript segments.
    """
    result = get_episode_summary(episode_id, base_url=BASE_URL)
    ep = result["episode"]
    chunks = result["chunks"]
    lines = [
        f"Episode: {ep['filename']}",
        f"Source: {ep.get('source_url') or 'local file'}",
        f"Total chunks: {ep['chunk_count']}",
        "",
        "Opening transcript:",
    ]
    for c in chunks:
        lines.append(f"  [{c['start_ts']:.1f}s] {c['text']}")
    return "\n".join(lines)


@mcp.tool()
def find_mentions_tool(topic: str, limit: int = 10) -> str:
    """Find all places in the library where a topic is mentioned.

    Args:
        topic: Topic or keyword to search for (e.g. 'fine-tuning', 'RAG', 'latency').
        limit: Maximum results to return (default 10).

    Returns:
        Formatted list of transcript chunks mentioning the topic, with timestamps.
    """
    limit = min(limit, 20)
    chunks = find_mentions(topic, base_url=BASE_URL, limit=limit)
    return format_chunks_for_mcp(chunks)


if __name__ == "__main__":
    mcp.run()
