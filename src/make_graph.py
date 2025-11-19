from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import SecretStr
import os

from .state import MultiState
from .utils import prepare_multimodal_message
from .prompts.multimodal_prompt import multimodal_prompt
from .models import get_multimodal_model

multimodal_agent = create_agent(
    model=get_multimodal_model(),
    tools=[],
    system_prompt=multimodal_prompt
)

async def multimodal_node(state: MultiState) -> Command[Literal["__end__"]]:   # after multimodal -> stop (could change later)
    """
    Handles multimodal inputs with multimodal model
    """

    # construct multimodal input message
    multimodal_msg = prepare_multimodal_message(state)  # returns HumanMessage

    # clear history of last message to swap last one with the new, multimodal one
    history = state.get("messages", [])[:-1] if state.get("messages", []) else []
    updated_history = history + [multimodal_msg]  # LG wants lists to concatenate messages

    result = await multimodal_agent.ainvoke({"messages": updated_history})
    last_msg = result["messages"][-1]

    return Command(
        update={
            "messages" : [last_msg],  # must be a list
            "images" : [],  # clearing images after invocation, keep memory lightweight
        },
        goto=END
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

    if save_display:
        # save the graph display to file
        img = graph.get_graph().draw_mermaid_png() # returns bytes
        # save the bytes to file 
        with open("./graph.png", "wb") as f:
            f.write(img)
        print("Graph display saved to ./src/graph.png")

    return graph