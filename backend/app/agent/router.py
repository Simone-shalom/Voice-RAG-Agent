import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from .graph import build_graph
from .streaming import stream_agent_response

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    reply: str
    steps: int


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
        graph = build_graph(db=db, llm=llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)], "step_count": 0},
            config={"configurable": {"thread_id": request.thread_id}},
        )
        last_msg = result["messages"][-1]
        return ChatResponse(reply=last_msg.content, steps=result["step_count"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        async for chunk in stream_agent_response(
            message=request.message,
            thread_id=request.thread_id,
            db=db,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
