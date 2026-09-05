from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from .tools import make_tools

MAX_STEPS = 6


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def build_graph(db: Session, llm):
    """
    Build and compile the LangGraph agent graph.

    db: SQLAlchemy session (captured in tool closures)
    llm: LangChain chat model (e.g. ChatAnthropic)

    Returns a compiled graph with MemorySaver checkpointing.
    """
    tools = make_tools(db)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    memory = MemorySaver()

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if state["step_count"] >= MAX_STEPS:
            return END
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=memory)
