from sqlalchemy import text
from sqlalchemy.orm import Session

from ..ingest.embedder import embed_texts


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
