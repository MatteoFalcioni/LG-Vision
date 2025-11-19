import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from make_graph import get_graph
from utils import get_multimodal_prompt, get_mpllry_b64
import uuid
from tqdm import tqdm

async def main():

    # memory (take this off if we want independent evaluations)
    checkpointer = InMemorySaver()
    
    # Define graph
    graph = get_graph(checkpointer)

    # Set user ID for storing memories
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    
    # TODO: get a mapillary image from api 
    images = get_mpllry_b64()
    
    # Construct multimodal system message
    sys_msg = get_multimodal_prompt() 

    print("\n=== Evaluating Mapillary images ===\n")

    for img_b64 in tqdm(images):
    
        # Initialize state with the image 
        init_state = {
            "messages": [sys_msg],
            "images": [img_b64]
        }
        
        # AI invocation
        print("\nAssistant: ", end="", flush=True)
        response = None
        async for chunk in graph.astream(init_state, stream_mode="values", config=config):
            response = chunk["messages"][-1]
        if response:
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main())