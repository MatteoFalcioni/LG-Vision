from langgraph.graph import START, StateGraph
from langgraph.types import Command
from typing import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .state import MultiState
from .utils import prepare_multimodal_message
from .prompts.mpllry_prompt import mpllry_prompt

mpllry_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt=mpllry_prompt
)

async def multimodal_node(state: MultiState) -> Command[Literal["__end__"]]:   # after multimodal -> stop 
    """
    Handles multimodal inputs with multimodal model
    """

    # construct multimodal input message
    multimodal_msg = prepare_multimodal_message(state)  # returns HumanMessage w/ content blocks w/ image

    # concatenate chat history with new multimodal message
    history = state.get("messages", []) if state.get("messages", []) else []
    updated_history = history + [multimodal_msg]  # LG wants lists to concatenate messages

    result = await mpllry_agent.ainvoke({"messages": updated_history})
    last_msg = result["messages"][-1]

    return Command(
        update={
            "messages" : [last_msg],  # must be a list
            "images" : [],  # clearing images after invocation, keep memory lightweight
        },
        goto="__end__"
    )

def get_graph(checkpointer, save_display=False) -> StateGraph:
    """
    Get the builder for the graph
    """
    builder = StateGraph(MultiState)
    # nodes
    builder.add_node("multimodal_agent", multimodal_node)
    # edges
    builder.add_edge(START, "multimodal_agent")

    graph = builder.compile(checkpointer=checkpointer)

    return graph