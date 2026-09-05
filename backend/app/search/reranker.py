import os
from functools import lru_cache

import cohere


@lru_cache(maxsize=1)
def _client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set — cannot use hybrid+rerank mode")
    return cohere.Client(api_key=api_key)


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Rerank chunks using Cohere Rerank API.
    Returns chunks sorted by Cohere's relevance score with 'similarity' updated.
    Raises RuntimeError if COHERE_API_KEY is not set.
    """
    if not chunks:
        return []
    texts = [c["text"] for c in chunks]
    response = _client().rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=min(top_n, len(chunks)),
    )
    return [
        {**chunks[r.index], "similarity": float(r.relevance_score)}
        for r in response.results
    ]
