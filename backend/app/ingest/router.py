import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .service import ingest_audio, ingest_from_url

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestURLRequest(BaseModel):
    url: str


class EpisodeResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int


@router.post("/upload", response_model=EpisodeResponse)
def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = Path(tmp.name)
        episode = ingest_audio(tmp_path, file.filename or "audio", None, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))


@router.post("/url", response_model=EpisodeResponse)
def ingest_url(body: IngestURLRequest, db: Session = Depends(get_db)):
    try:
        episode = ingest_from_url(body.url, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EpisodeResponse(id=str(episode.id), filename=episode.filename, chunk_count=len(episode.chunks))
