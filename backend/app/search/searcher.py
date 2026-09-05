from sqlalchemy import text
from sqlalchemy.orm import Session

from ..ingest.embedder import embed_texts
from .bm25 import bm25_search
from .reranker import rerank
from .rrf import reciprocal_rank_fusion


def semantic_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Embed the query and find the nearest chunks by cosine similarity.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}.
    """
    embedding = embed_texts([query])[0]
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text,
                   1 - (c.embedding <=> :emb::vector) AS similarity
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> :emb::vector
            LIMIT :limit
            """
        ),
        {"emb": embedding_str, "limit": limit},
    ).fetchall()

    return [
        {
            "chunk_id": str(r.id),
            "episode_id": str(r.episode_id),
            "start_ts": r.start_ts,
            "end_ts": r.end_ts,
            "text": r.text,
            "similarity": float(r.similarity),
        }
        for r in rows
    ]


def hybrid_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Fuse semantic and BM25 results via Reciprocal Rank Fusion.
    Fetches 2× limit from each sub-searcher for fusion headroom.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}
    where similarity is the RRF score.
    """
    fetch = limit * 2
    sem = semantic_search(query, limit=fetch, db=db)
    bm25 = bm25_search(query, limit=fetch, db=db)
    fused = reciprocal_rank_fusion([sem, bm25])
    return [
        {**chunk, "similarity": chunk["rrf_score"]}
        for chunk in fused[:limit]
    ]


def hybrid_rerank_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Hybrid RRF search followed by Cohere reranking.
    Raises RuntimeError if COHERE_API_KEY is not set.
    """
    candidates = hybrid_search(query, limit=limit * 2, db=db)
    return rerank(query, candidates, top_n=limit)
