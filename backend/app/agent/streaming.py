# backend/app/agent/streaming.py
import asyncio
import base64
import json
import os
from typing import AsyncGenerator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from .chunker import SentenceChunker
from .graph import build_graph
from ..voice.tts import synthesise


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_agent_response(
    message: str,
    thread_id: str,
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE strings.

    Event types:
      {"type": "token",  "text": "..."}                          — each streamed token
      {"type": "audio",  "data": "<b64>", "text": "..."}          — synthesised sentence chunk, in playback order
      {"type": "done"}                                            — end of stream
      {"type": "error", "text": "..."}                            — fatal error (missing API key, or a failure mid-stream)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield _sse({"type": "error", "text": "ANTHROPIC_API_KEY not set"})
        return

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    graph = build_graph(db=db, llm=llm)
    chunker = SentenceChunker()

    async def _tts_chunk(text: str) -> str:
        loop = asyncio.get_running_loop()
        mp3_bytes = await loop.run_in_executor(None, synthesise, text)
        b64 = base64.b64encode(mp3_bytes).decode()
        return _sse({"type": "audio", "data": b64, "text": text})

    tts_tasks: list[asyncio.Task] = []

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)], "step_count": 0},
            config={"configurable": {"thread_id": thread_id}},
            version="v1",
        ):
            if event["event"] != "on_chat_model_stream":
                continue
            token = event["data"]["chunk"].content
            if not isinstance(token, str) or not token:
                continue

            yield _sse({"type": "token", "text": token})

            for sentence in chunker.push(token):
                tts_tasks.append(asyncio.create_task(_tts_chunk(sentence)))

        for sentence in chunker.flush():
            tts_tasks.append(asyncio.create_task(_tts_chunk(sentence)))

        for task in tts_tasks:
            yield await task

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})

    finally:
        for task in tts_tasks:
            if not task.done():
                task.cancel()
