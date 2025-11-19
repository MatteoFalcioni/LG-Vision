from langchain_core.messages import HumanMessage
from state import MultiState
from prompts.mpllry_prompt import prompt
import base64

# this is actually used in main.py
def encode_b64_from_path(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_b64_paths(file_paths: list[str]) -> list[str]:
    return [encode_b64_from_path(file_path) for file_path in file_paths]

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
        text = "Analyze this image"
    
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

def get_multimodal_prompt(good_imgs_paths : list[str], bad_imgs_paths : list[str], text : str = prompt):
    """
    Constructs a multimodal system message, given the textual prompt and images to refer to. 
    The textual prompt defaults to our own custom system prompt.
    """
    content = [[{"type": "text", "text": text}]]  # start with the prompt

    # encode both good and bad images
    good_b64 = encode_b64_paths(good_imgs_paths)
    bad_b64  = encode_b64_paths(bad_imgs_paths)

    # add good images
    for good_img in good_b64:
        good_text = "This image is acceptable:"
        content.append(
            {"type" : "text", "text" : good_text},
            {
                "type" : "image",
                "base64" : good_img,
                "mime_type" : "image/png"  # or image/jpg if jpg
            }
        )
    # add bad images
    for bad_img in bad_b64:
        bad_text = "This image is discardable:"
        content.append(
            {"type" : "text", "text" : bad_text},
            {
                "type" : "image",
                "base64" : bad_img,
                "mime_type" : "image/png"  # or image/jpg if jpg
            }
        )

    system_prompt = HumanMessage(content_blocks=content)
    return system_prompt


def get_mpllry_b64(num_imgs : int) -> list:
    """
    Leverages the Mapillary API to download `num_imgs` images.
    Encodes the images in base 64, and returns a list of the encodings
    """



