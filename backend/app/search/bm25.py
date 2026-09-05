from sqlalchemy import text
from sqlalchemy.orm import Session


def bm25_search(query: str, limit: int, db: Session) -> list[dict]:
    """
    Full-text search using PostgreSQL tsvector/ts_rank.
    Returns list of {chunk_id, episode_id, start_ts, end_ts, text, similarity}
    where similarity is the ts_rank score (higher = more relevant).
    """
    rows = db.execute(
        text(
            """
            SELECT c.id, c.episode_id, c.start_ts, c.end_ts, c.text,
                   ts_rank(c.text_search, query) AS bm25_score
            FROM chunks c,
                 plainto_tsquery('english', :query_text) query
            WHERE c.text_search @@ query
            ORDER BY bm25_score DESC
            LIMIT :limit
            """
        ),
        {"query_text": query, "limit": limit},
    ).fetchall()

    return [
        {
            "chunk_id": str(r.id),
            "episode_id": str(r.episode_id),
            "start_ts": r.start_ts,
            "end_ts": r.end_ts,
            "text": r.text,
            "similarity": float(r.bm25_score),
        }
        for r in rows
    ]
