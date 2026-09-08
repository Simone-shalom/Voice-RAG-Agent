import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..db.models import ChatMessage, ChatThread

logger = logging.getLogger(__name__)

TITLE_MAX_LEN = 60


def _make_title(first_message: str) -> str:
    text = " ".join(first_message.split())  # collapse whitespace/newlines
    if not text:
        return "New conversation"
    if len(text) <= TITLE_MAX_LEN:
        return text
    return text[:TITLE_MAX_LEN].rstrip() + "…"


def ensure_thread(db: Session, thread_id: str, first_message: str) -> None:
    """Creates the thread row on its first message; no-ops if it already exists."""
    if db.get(ChatThread, thread_id) is not None:
        return
    db.add(ChatThread(id=thread_id, title=_make_title(first_message)))
    db.commit()


def save_message(
    db: Session,
    thread_id: str,
    role: str,
    text: str,
    sources: list[dict] | None = None,
) -> None:
    db.add(
        ChatMessage(
            id=uuid.uuid4(),
            thread_id=thread_id,
            role=role,
            text=text,
            sources=sources,
        )
    )
    thread = db.get(ChatThread, thread_id)
    if thread is not None:
        thread.updated_at = datetime.now(timezone.utc)
    db.commit()


def list_threads(db: Session) -> list[ChatThread]:
    return db.query(ChatThread).order_by(ChatThread.updated_at.desc()).all()


def get_thread(db: Session, thread_id: str) -> ChatThread | None:
    return db.get(ChatThread, thread_id)
