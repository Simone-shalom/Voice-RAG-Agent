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


MAX_SOURCES = 8  # cap how many chunks are surfaced as citations per answer


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def stream_agent_response(
    message: str,
    thread_id: str,
    db: Session,
    mode: str = "hybrid",
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE strings.

    Event types: token, audio (b64+text), sources, done, error.
      {"type": "token",  "text": "..."}                          — each streamed token
      {"type": "audio",  "data": "<b64>", "text": "..."}          — synthesised sentence chunk, in playback order
      {"type": "sources", "data": [{"episode_id","start_ts","end_ts","text"}, ...]}
                                                                   — deduplicated chunks the agent actually
                                                                     retrieved, sent once before "done";
                                                                     omitted entirely if nothing was retrieved
      {"type": "done"}                                            — end of stream
      {"type": "error", "text": "..."}                            — fatal error (missing API key, or a failure mid-stream)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield _sse({"type": "error", "text": "ANTHROPIC_API_KEY not set"})
        return

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    sources_sink: list[dict] = []
    graph = build_graph(db=db, llm=llm, mode=mode, sources_sink=sources_sink)
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

        if sources_sink:
            seen = set()
            deduped: list[dict] = []
            for src in sources_sink:
                key = (src.get("episode_id"), src.get("start_ts"), src.get("end_ts"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append({
                    "episode_id": src.get("episode_id"),
                    "start_ts": src.get("start_ts"),
                    "end_ts": src.get("end_ts"),
                    "text": src.get("text"),
                })
                if len(deduped) >= MAX_SOURCES:
                    break
            yield _sse({"type": "sources", "data": deduped})

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})

    finally:
        for task in tts_tasks:
            if not task.done():
                task.cancel()
