from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import SecretStr
import os


def get_multimodal_model():
    """
    Get the multimodal model
    """
    load_dotenv()

    provider = os.getenv("PROVIDER", "QWEN")
    
    if provider == "QWEN":
        model_alias = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
        multimodal_model = ChatOpenAI(
            api_key=SecretStr(os.environ["FIREWORKS_API_KEY"]),
            base_url="https://api.fireworks.ai/inference/v1",
            model=model_alias
        )
    elif provider == "OPENAI":
        print("Using OPENAI model")
        model_alias = "gpt-4o"
        multimodal_model=ChatOpenAI(
            model=model_alias,
            api_key=SecretStr(os.environ["OPENAI_API_KEY"])
        )
    elif provider == "ANTHROPIC":
        print("Using ANTHROPIC model")
        model_alias = "claude-sonnet-4-5"
        multimodal_model=ChatAnthropic(
            model=model_alias,
            api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"])
        )
    else:
        raise RuntimeError(f"Invalid provider: {provider}")

    return multimodal_model