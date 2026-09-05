import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph


def make_db():
    return MagicMock()


def make_sequential_llm(*responses):
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = list(responses)
    return mock


def test_simple_question_uses_one_llm_call():
    """
    A factual question the LLM can answer without tools should result in:
    - 1 LLM call (no tool calls in the response)
    - step_count == 1
    DoD requirement: simple query takes fewer steps than complex query.
    """
    mock_llm = make_sequential_llm(
        AIMessage(content="The capital of France is Paris.")
    )
    graph = build_graph(db=make_db(), llm=mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")], "step_count": 0},
        config={"configurable": {"thread_id": "integration-simple"}},
    )

    assert mock_llm.invoke.call_count == 1
    assert result["step_count"] == 1
    assert "Paris" in result["messages"][-1].content


def test_multi_hop_question_uses_multiple_llm_calls():
    """
    A question that needs transcript search results in:
    - First LLM call → tool call (search_transcripts)
    - Tool executes → returns results
    - Second LLM call → final answer
    - step_count == 2
    DoD requirement: complex query takes more steps than simple query.
    """
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_abc",
            "name": "search_transcripts",
            "args": {"query": "neural networks introduction", "limit": 5},
            "type": "tool_call",
        }],
    )
    final_response = AIMessage(
        content="Neural networks were introduced at 5.2s in episode ep1."
    )
    mock_llm = make_sequential_llm(tool_call_response, final_response)

    with patch("app.agent.tools.hybrid_search", return_value=[
        {"chunk_id": "c1", "episode_id": "ep1", "start_ts": 5.2, "end_ts": 15.0,
         "text": "an introduction to neural networks", "similarity": 0.95}
    ]):
        graph = build_graph(db=make_db(), llm=mock_llm)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Where are neural networks discussed?")],
             "step_count": 0},
            config={"configurable": {"thread_id": "integration-multi"}},
        )

    assert mock_llm.invoke.call_count == 2
    assert result["step_count"] == 2
    assert "5.2s" in result["messages"][-1].content


def test_simple_step_count_less_than_multi_hop():
    """Explicit DoD assertion: simple question produces fewer steps than multi-hop."""
    simple_llm = make_sequential_llm(AIMessage(content="Direct answer."))
    simple_graph = build_graph(db=make_db(), llm=simple_llm)
    simple_result = simple_graph.invoke(
        {"messages": [HumanMessage(content="Simple question")], "step_count": 0},
        config={"configurable": {"thread_id": "dod-simple"}},
    )

    multi_llm = make_sequential_llm(
        AIMessage(content="", tool_calls=[{
            "id": "c1", "name": "search_transcripts",
            "args": {"query": "topic"}, "type": "tool_call",
        }]),
        AIMessage(content="Found at 10.0s."),
    )
    with patch("app.agent.tools.hybrid_search", return_value=[]):
        multi_graph = build_graph(db=make_db(), llm=multi_llm)
        multi_result = multi_graph.invoke(
            {"messages": [HumanMessage(content="Complex multi-hop question")], "step_count": 0},
            config={"configurable": {"thread_id": "dod-multi"}},
        )

    assert simple_result["step_count"] < multi_result["step_count"], (
        f"Simple ({simple_result['step_count']}) should be < multi ({multi_result['step_count']})"
    )
