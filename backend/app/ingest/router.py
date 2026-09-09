import logging
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from openai import APIError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.auth import require_api_key
from ..core.rate_limit import limiter
from ..db.session import get_db
from .service import ingest_audio, ingest_from_rss, ingest_from_url
from .validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    summary: str | None = None
    chapters: list[dict] | None = None


class IngestRSSRequest(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=20)


class RSSErrorItem(BaseModel):
    title: str
    error: str


class RSSIngestResponse(BaseModel):
    episodes: list[EpisodeResponse]
    errors: list[RSSErrorItem]


@router.post("/upload", response_model=EpisodeResponse)
@limiter.limit("10/minute")
def upload_audio(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        data = read_capped(file.file, MAX_UPLOAD_BYTES)
        if not is_probably_audio(data[:16]):
            raise ValueError("file does not look like a supported audio format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except APIError:
        raise HTTPException(status_code=400, detail="audio transcription failed")
    except Exception:
        logger.exception("ingest upload failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return EpisodeResponse(
        id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks),
        summary=episode.summary, chapters=episode.chapters,
    )


@router.post("/url", response_model=EpisodeResponse)
@limiter.limit("10/minute")
def ingest_url(request: Request, body: IngestURLRequest, db: Session = Depends(get_db)):
    try:
        episode = ingest_from_url(body.url, db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPError:
        raise HTTPException(status_code=400, detail="could not fetch the URL")
    except Exception:
        logger.exception("ingest from URL failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return EpisodeResponse(
        id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks),
        summary=episode.summary, chapters=episode.chapters,
    )


@router.post("/rss", response_model=RSSIngestResponse)
@limiter.limit("5/minute")
def ingest_rss(request: Request, body: IngestRSSRequest, db: Session = Depends(get_db)):
    try:
        result = ingest_from_rss(body.url, db, limit=body.limit)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("ingest from RSS failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")
    return RSSIngestResponse(
        episodes=[
            EpisodeResponse(
                id=str(ep.id), filename=ep.filename, chunk_count=len(ep.chunks),
                summary=ep.summary, chapters=ep.chapters,
            )
            for ep in result["episodes"]
        ],
        errors=[RSSErrorItem(**err) for err in result["errors"]],
    )
