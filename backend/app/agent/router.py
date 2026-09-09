import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.rate_limit import limiter
from ..db.session import get_db
from .graph import build_graph
from .history import delete_thread, ensure_thread, get_thread, list_threads, save_message
from .streaming import stream_agent_response, _sse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

AgentSearchMode = Literal["semantic", "bm25", "hybrid", "hybrid+rerank"]


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    mode: AgentSearchMode = "hybrid"


class ChatResponse(BaseModel):
    reply: str
    steps: int


class ThreadSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ThreadMessage(BaseModel):
    role: str
    text: str
    sources: list[dict] | None = None
    created_at: str


class ThreadDetail(BaseModel):
    id: str
    title: str
    messages: list[ThreadMessage]


def _save_message_safely(db: Session, thread_id: str, role: str, text: str, sources: list[dict] | None = None) -> None:
    # History is a convenience feature — a persistence hiccup must never
    # break the live chat/voice response the user is actually waiting on.
    try:
        save_message(db, thread_id, role, text, sources)
    except Exception:
        logger.exception("failed to persist chat message for thread %s", thread_id)


def _ensure_thread_safely(db: Session, thread_id: str, first_message: str) -> None:
    try:
        ensure_thread(db, thread_id, first_message)
    except Exception:
        logger.exception("failed to create chat thread %s", thread_id)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        _ensure_thread_safely(db, body.thread_id, body.message)
        _save_message_safely(db, body.thread_id, "user", body.message)

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=body.message)], "step_count": 0},
            config={"configurable": {"thread_id": body.thread_id}},
        )
        last_msg = result["messages"][-1]
        _save_message_safely(db, body.thread_id, "assistant", last_msg.content)
        return ChatResponse(reply=last_msg.content, steps=result["step_count"])
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("agent chat failed")
        raise HTTPException(status_code=500, detail="Internal error — please try again later.")


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for event in stream_agent_response(
            message=body.message,
            thread_id=body.thread_id,
            db=db,
            mode=body.mode,
        ):
            yield _sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/threads", response_model=list[ThreadSummary])
def get_threads(db: Session = Depends(get_db)):
    return [
        ThreadSummary(id=t.id, title=t.title, updated_at=t.updated_at.isoformat())
        for t in list_threads(db)
    ]


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
def get_thread_detail(thread_id: str, db: Session = Depends(get_db)):
    thread = get_thread(db, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return ThreadDetail(
        id=thread.id,
        title=thread.title,
        messages=[
            ThreadMessage(role=m.role, text=m.text, sources=m.sources, created_at=m.created_at.isoformat())
            for m in thread.messages
        ],
    )


@router.delete("/threads/{thread_id}", status_code=204)
def delete_thread_endpoint(thread_id: str, db: Session = Depends(get_db)):
    if not delete_thread(db, thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
