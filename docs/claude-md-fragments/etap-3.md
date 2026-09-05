# CLAUDE.md fragment — Etap 3: LangGraph Agent

## Agent endpoint

`POST /agent/chat` — accepts `{message: str, thread_id: str}`, returns `{reply: str, steps: int}`.

Requires `ANTHROPIC_API_KEY` in environment. Missing key → HTTP 400.

```bash
curl -X POST localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is discussed at minute 5?", "thread_id": "session-1"}'
# Returns: {"reply": "...", "steps": 2}
```

## Agent tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `search_transcripts` | `query: str, limit: int = 5` | Hybrid search across all episodes |
| `get_context_around_timestamp` | `episode_id: str, timestamp: float, window_seconds: float = 30.0` | Chunks near a timestamp |
| `compare_across_episodes` | `query: str, episode_ids: str, limit_per_episode: int = 3` | Per-episode grouped results |
| `summarise_segment` | `text: str` | Formats a segment for LLM analysis |

To add a new tool: define `@tool` inside `make_tools(db)` in `backend/app/agent/tools.py`, following the closure pattern for db access.

## Graph architecture

```
[START] → agent_node → should_continue → tools_node → agent_node → ...
                                       ↘ END (no tool_calls or step_count >= MAX_STEPS)
```

- `MAX_STEPS = 6` in `backend/app/agent/graph.py` — increase with care (each step = 1 LLM call)
- `MemorySaver` persists conversation per `thread_id` in-memory (restarts wipe history)
- State: `{messages: list[BaseMessage], step_count: int}`

## DoD verification

The integration test in `test_agent_integration.py` verifies the agent is a real conditional graph:
- `test_simple_question_uses_one_llm_call`: step_count == 1
- `test_multi_hop_question_uses_multiple_llm_calls`: step_count == 2
- `test_simple_step_count_less_than_multi_hop`: explicit comparison

## Decisions made (see DECISIONS.md for full rationale)

- LangGraph StateGraph over LangChain AgentExecutor (conditional routing, not deprecated)
- Claude Haiku over GPT-4o-mini (fastest Claude with tool use support)
- ANTHROPIC_API_KEY checked per-request, not at startup (app works without key when agent not used)
