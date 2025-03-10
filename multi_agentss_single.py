import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv

# Enable nested event loops (useful for notebooks)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError

# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

async def test_model(llm, query: str) -> str:
    """Send a query to the given language model and print its response."""
    user_message = UserMessage(content=query)
    try:
        response = await asyncio.wait_for(llm.create({"messages": [user_message]}), timeout=30)
        result_text = response.get_text_content()
        print(f"Response from {llm.__class__.__name__}: {result_text}")
        return result_text
    except Exception as e:
        print(f"Error testing model {llm.__class__.__name__}: {e}")
        return ""

async def main() -> None:
    # Test Watsonx model
    print("Testing Watsonx model:")
    llm_watsonx = await WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    await test_model(llm_watsonx, "What is the capital of Italy?")

    # Test Ollama model
    print("\nTesting Ollama model:")
    llm_ollama = await ChatModel.from_name("ollama:granite3.1-dense:8b")
    await test_model(llm_ollama, "What is the capital of Italy?")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))
