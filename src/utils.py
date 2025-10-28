from langchain_core.messages import HumanMessage
from .state import MultiState
import base64

# this is actually used in main.py
def encode_b64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def prepare_multimodal_message(state: MultiState) -> HumanMessage:
    """
    Helper to create multimodal message from state

    Returns:
        message (HumanMessage): The multimodal message
    """
    messages = state.get("messages", [])
    
    # Get text from last message or use default
    if messages and isinstance(messages[-1].content, str):
        text = messages[-1].content
    else:
        text = "Analyze this image based on the prompt provided"
    
    content_blocks = [{"type": "text", "text": text}]   # it is a list of typed dicts, see https://docs.langchain.com/oss/python/langchain/messages#multimodal
    
    # Add images
    for img_b64 in state.get("images", []):
        content_blocks.append({
            "type": "image",
            "base64": img_b64,
            "mime_type": "image/png"  # it would be better to auto detect myme at runtime - so maybe img encoding should be here and not in main
        })


    # construct the messages as HumanMessage(content_blocks=...)
    message = HumanMessage(content_blocks=content_blocks)  # v1 format, see https://docs.langchain.com/oss/python/langchain/messages#multimodal
    
    return message   # NOTE: returns msg as is, then you need to wrap it in a list!