from langgraph.graph import START, StateGraph
from langgraph.types import Command
from typing import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from state import MultiState
from utils import prepare_multimodal_message

load_dotenv()

# Structured output
class BinaryOutput(BaseModel):
    response: Literal["yes", "no"]

# make its output structured: yes/no
mpllry_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt="You are an AI assistant that evaluates the quality of Mapilary images",  # short prompt because the real one is passed at runtime
    state_schema=MultiState,
    response_format=BinaryOutput
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