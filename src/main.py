import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from make_graph import get_graph
from utils import encode_b64
import uuid


async def main():
    
    load_keys = load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("FIREWORKS_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("no api key in environment!")
    print(f"Loaded env: {load_keys}")

    # memory
    checkpointer = InMemorySaver()
    
    # Define graph
    graph = get_graph(checkpointer)

    # Set user ID for storing memories
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    print("***EXAMPLE 1***")
    ex_img_b64 = encode_b64("../examples/ortofoto.png")

    init_state1 = {
        "messages" : [HumanMessage(content="What can you see in this image?")],
        "images" : [ex_img_b64]
    }

    async for chunk in graph.astream(init_state1, stream_mode="values", config=config):
        chunk["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())