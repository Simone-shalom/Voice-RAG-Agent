from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db.session import get_db

router = APIRouter(prefix="/episodes", tags=["episodes"])


class EpisodeSummary(BaseModel):
    id: str
    filename: str
    source_url: str | None
    chunk_count: int
    summary: str | None = None
    chapters: list[dict] | None = None


class ChunkItem(BaseModel):
    id: str
    start_ts: float
    end_ts: float
    text: str
    speaker_label: str | None = None


@router.get("", response_model=list[EpisodeSummary])
def list_episodes(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT e.id, e.filename, e.source_url, e.summary, e.chapters, COUNT(c.id) AS chunk_count
        FROM episodes e
        LEFT JOIN chunks c ON c.episode_id = e.id
        GROUP BY e.id
        ORDER BY e.created_at DESC
    """)).fetchall()
    return [
        EpisodeSummary(
            id=str(r.id),
            filename=r.filename,
            source_url=r.source_url,
            chunk_count=r.chunk_count,
            summary=r.summary,
            chapters=r.chapters,
        )
        for r in rows
    ]


@router.get("/{episode_id}", response_model=EpisodeSummary)
def get_episode(episode_id: str, db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT e.id, e.filename, e.source_url, e.summary, e.chapters, COUNT(c.id) AS chunk_count
            FROM episodes e
            LEFT JOIN chunks c ON c.episode_id = e.id
            WHERE e.id = :ep_id
            GROUP BY e.id
        """),
        {"ep_id": episode_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Episode not found")
    return EpisodeSummary(
        id=str(row.id),
        filename=row.filename,
        source_url=row.source_url,
        chunk_count=row.chunk_count,
        summary=row.summary,
        chapters=row.chapters,
    )


@router.get("/{episode_id}/chunks", response_model=list[ChunkItem])
def get_episode_chunks(
    episode_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text("""
            SELECT id, start_ts, end_ts, text, speaker_label
            FROM chunks
            WHERE episode_id = :ep_id
            ORDER BY start_ts ASC
            LIMIT :limit
        """),
        {"ep_id": episode_id, "limit": limit},
    ).fetchall()
    return [
        ChunkItem(id=str(r.id), start_ts=r.start_ts, end_ts=r.end_ts, text=r.text, speaker_label=r.speaker_label)
        for r in rows
    ]
