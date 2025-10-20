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

    # memory
    checkpointer = InMemorySaver()
    
    # Define graph
    graph = get_graph(checkpointer)

    # Set user ID for storing memories
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    # Get image path from user
    print("\n=== Vision AI Chat ===")
    print("Type '/bye' at any time to exit\n")
    
    while True:
        img_path = input("Enter an image path: ").strip()
        
        if img_path.lower() == "/bye":
            print("Goodbye!")
            return
        
        if not os.path.exists(img_path):
            print(f"Error: Image path '{img_path}' does not exist. Please try again.\n")
            continue
        
        try:
            img_b64 = encode_b64(img_path)
            break
        except Exception as e:
            print(f"Error loading image: {e}\n")
            continue
    
    # Get initial message
    user_message = input("\nYou: ").strip()
    if user_message.lower() == "/bye":
        print("Goodbye!")
        return
    
    # Initialize state with first message and image
    init_state = {
        "messages": [HumanMessage(content=user_message)],
        "images": [img_b64]
    }
    
    # First interaction
    print("\nAssistant: ", end="", flush=True)
    response = None
    async for chunk in graph.astream(init_state, stream_mode="values", config=config):
        response = chunk["messages"][-1]
    if response:
        print(response.content)
    
    # Conversation loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "/bye":
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Continue conversation with new message
        state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        print("\nAssistant: ", end="", flush=True)
        response = None
        async for chunk in graph.astream(state, stream_mode="values", config=config):
            response = chunk["messages"][-1]
        if response:
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main())