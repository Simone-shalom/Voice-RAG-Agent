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
from .streaming import stream_agent_response

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


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=body.message)], "step_count": 0},
            config={"configurable": {"thread_id": body.thread_id}},
        )
        last_msg = result["messages"][-1]
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
        async for chunk in stream_agent_response(
            message=body.message,
            thread_id=body.thread_id,
            db=db,
            mode=body.mode,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
