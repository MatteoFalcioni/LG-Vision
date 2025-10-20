from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import SecretStr
import os

from state import MultiState
from utils import prepare_multimodal_message
from prompts.multimodal_prompt import multimodal_prompt

load_dotenv()

provider = os.getenv("PROVIDER", "QWEN")

if provider == "QWEN":
    if os.getenv("MODEL_DIM") == "SMALL":
        print("Using QWEN 32b model")
        model_alias = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
    elif os.getenv("MODEL_DIM") == "LARGE":
        print("Using QWEN 235b model")
        model_alias = "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct"
    else:
        raise RuntimeError(f"Invalid model dimension: {os.getenv('MODEL_DIM')}")
    multimodal_model = ChatOpenAI(
        api_key=SecretStr(os.environ["FIREWORKS_API_KEY"]),
        base_url="https://api.fireworks.ai/inference/v1",
        model=model_alias
    )
elif provider == "OPENAI":
    print("Using OPENAI model")
    model_alias = "gpt-5"
    multimodal_model=ChatOpenAI(
        model=model_alias,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )
elif provider == "ANTHROPIC":
    print("Using ANTHROPIC model")
    model_alias = "claude-sonnet-4-5"
    multimodal_model=ChatAnthropic(
        model=model_alias,
        api_key=SecretStr(os.getenviron["ANTHROPIC_API_KEY"])
    )
else:
    raise RuntimeError(f"Invalid provider: {provider}")


multimodal_agent = create_agent(
    model=multimodal_model,
    tools=[],
    prompt=multimodal_prompt
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
            #"images" : [],  # not clearing images after invocation - trying to preserve them in memory 
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