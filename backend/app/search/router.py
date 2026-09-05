import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .bm25 import bm25_search
from .searcher import hybrid_rerank_search, hybrid_search, semantic_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

SearchMode = Literal["semantic", "bm25", "hybrid", "hybrid+rerank"]


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
    mode: SearchMode = Query("semantic"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    try:
        if mode == "semantic":
            return semantic_search(query=q, limit=limit, db=db)
        elif mode == "bm25":
            return bm25_search(query=q, limit=limit, db=db)
        elif mode == "hybrid":
            return hybrid_search(query=q, limit=limit, db=db)
        else:  # hybrid+rerank
            return hybrid_rerank_search(query=q, limit=limit, db=db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
