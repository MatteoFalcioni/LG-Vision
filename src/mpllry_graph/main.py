import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from make_graph import get_graph
from utils import get_multimodal_prompt, get_mpllry_b64
from langchain_core.messages import HumanMessage
import uuid
from tqdm import tqdm

async def main():

    # memory (take this off if we want independent evaluations)
    checkpointer = InMemorySaver()
    
    # Define graph
    graph = get_graph(checkpointer)

    # Set thread ID for memory
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}
    
    # get a mapillary image from api 
    images = get_mpllry_b64(
        num_points=2,
        save_images=True,
        save_folder="images"
        )

    # Construct multimodal system message
    # sys_msg = get_multimodal_prompt() 

    print("\n=== Evaluating Mapillary images ===\n")

    for img_b64 in tqdm(images):
    
        # Initialize state with the image 
        init_state = {
            "messages": [HumanMessage(content="analize this image")],
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