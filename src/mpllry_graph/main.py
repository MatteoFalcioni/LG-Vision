import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from make_graph import get_graph
from utils import get_multimodal_prompt, get_mpllry_b64
import uuid
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import HumanMessage
import base64
async def main():

    load_dotenv()

    # memory (take this off if we want independent evaluations)
    # checkpointer = InMemorySaver()
    
    # Define graph
    graph = get_graph()

    # Set thread ID for memory
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}
    
    '''# get a mapillary image from api 
    images = get_mpllry_b64(
        num_points=1,
        save_images=True,
        save_folder=f"images/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    if len(images) == 0:
        print("No images found. Stopping execution.")
        return   # stop execution if no images are found'''

    path = "images/run_20251124_154556/1182020198891070.jpg"
    #encode image to base64
    with open(path, "rb") as image_file:
        images = [base64.b64encode(image_file.read()).decode('utf-8')]

    # Construct multimodal system message
    # 1) get paths of the examples (good = acceptable images, bad = discardable images) 
    good_dir = Path("./examples/good_examples/")
    bad_dir = Path("./examples/bad_examples/")
    good_paths = [str(p) for p in good_dir.iterdir() if p.is_file()]
    bad_paths = [str(p) for p in bad_dir.iterdir() if p.is_file()]
    # 2) construct the prompt (adds the example to the textual prompt in prompts/mpllry_prompt)
    sys_msg = get_multimodal_prompt(
        good_imgs_paths=good_paths,
        bad_imgs_paths=bad_paths,
        ) 

    print("\n=== Evaluating Mapillary images ===\n")

    for img_b64 in images:
    
        # Initialize state with the image 
        init_state = {
            "messages": [sys_msg] + [HumanMessage("Analize the following images")],  # NOTE: 
            "images": [img_b64]
        }
        
        # AI invocation
        print("\nAssistant: ", end="", flush=True)
        response = None
        async for chunk in graph.astream(init_state, stream_mode="values", config=config):
            response = chunk["messages"][-1]
        if response is not None:
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main())