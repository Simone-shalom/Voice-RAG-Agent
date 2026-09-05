import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph, MAX_STEPS


def make_db():
    return MagicMock()


def make_mock_llm(responses: list):
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = responses
    return mock


def test_build_graph_returns_compiled_graph():
    mock_llm = make_mock_llm([AIMessage(content="hello")])
    graph = build_graph(db=make_db(), llm=mock_llm)
    assert graph is not None


def test_max_steps_constant_exists():
    assert isinstance(MAX_STEPS, int)
    assert MAX_STEPS >= 3


def test_simple_query_ends_without_tool_call():
    """A direct answer (no tool_calls) ends the graph in one agent step."""
    mock_llm = make_mock_llm([
        AIMessage(content="The answer is 42.")
    ])
    graph = build_graph(db=make_db(), llm=mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 6 times 7?")], "step_count": 0},
        config={"configurable": {"thread_id": "test-simple"}},
    )

    assert mock_llm.invoke.call_count == 1
    assert result["messages"][-1].content == "The answer is 42."


def test_multi_step_query_calls_tools_then_answers():
    """Query that requires a tool call: agent calls tool, then answers."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "search_transcripts",
            "args": {"query": "machine learning", "limit": 5},
            "type": "tool_call",
        }],
    )
    final_answer = AIMessage(content="Machine learning was discussed at 12.3s.")
    mock_llm = make_mock_llm([tool_call_msg, final_answer])

    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 12.3, "end_ts": 18.0,
         "text": "machine learning basics", "similarity": 0.9}
    ]):
        graph = build_graph(db=make_db(), llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Where is ML discussed?")], "step_count": 0},
            config={"configurable": {"thread_id": "test-multi"}},
        )

    assert mock_llm.invoke.call_count == 2
    assert result["messages"][-1].content == "Machine learning was discussed at 12.3s."


def test_step_limit_prevents_infinite_loop():
    """When step_count reaches MAX_STEPS, graph must stop even if LLM wants more tools."""
    always_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_x",
            "name": "search_transcripts",
            "args": {"query": "loop"},
            "type": "tool_call",
        }],
    )
    mock_llm = make_mock_llm([always_tool_call] * (MAX_STEPS + 5))

    with patch("app.agent.tools.hybrid_search", return_value=[]):
        graph = build_graph(db=make_db(), llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Loop forever")], "step_count": 0},
            config={"configurable": {"thread_id": "test-limit"}},
        )

    assert mock_llm.invoke.call_count <= MAX_STEPS
