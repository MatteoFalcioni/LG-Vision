from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pydantic import SecretStr
import os

from state import MultiState
from utils import prepare_multimodal_message
from prompts.multimodal import multimodal_prompt

load_dotenv()

model_alias = os.getenv("MODEL_ALIAS", "accounts/fireworks/models/qwen2p5-vl-32b-instruct")

multimodal_model = ChatOpenAI(
    api_key=SecretStr(os.environ["FIREWORKS_API_KEY"]),
    base_url="https://api.fireworks.ai/inference/v1",
    model=model_alias,
    streaming=True,
)
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
            #"images" : [],  # not clearing images audios after invocation - trying to preserve them in memory 
        },
        goto=END
    )