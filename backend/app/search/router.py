from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .searcher import semantic_search

router = APIRouter(prefix="/search", tags=["search"])


class SearchResult(BaseModel):
    chunk_id: str
    episode_id: str
    start_ts: float
    end_ts: float
    text: str
    similarity: float


@router.get("", response_model=list[SearchResult])
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    try:
        return semantic_search(query=q, limit=limit, db=db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
