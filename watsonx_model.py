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
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError

# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

async def evaluate_model(llm, query: str) -> str:
    """
    Evaluates a given language model with a query and returns the text response.

    Args:
        llm: The language model instance to evaluate.
        query: The query string to send to the model.

    Returns:
        str: The text content of the model's response, or an empty string in case of error.
    """
    user_message = UserMessage(content=query)
    try:
        response = await asyncio.wait_for(llm.create({"messages": [user_message]}), timeout=30)
        return response.get_text_content()
    except Exception as e:
        print(f"Error evaluating model {llm.__class__.__name__}: {e}")
        return ""

async def main() -> None:
    """Main function to demonstrate Watsonx model evaluation."""
    try:
        print("Initializing Watsonx model...")
        llm_watsonx = await WatsonxChatModel.from_name(
            "watsonx:ibm/granite-3-8b-instruct",
            options={
                "project_id": WATSONX_PROJECT_ID,
                "api_key": WATSONX_API_KEY,
                "api_base": WATSONX_API_URL,
            },
        )
        print("Watsonx model initialized.")

        query = "What is the capital of Spain?"
        print(f"\nSending query to Watsonx model: '{query}'")
        response_text = await evaluate_model(llm_watsonx, query)

        if response_text:
            print(f"Watsonx Response: {response_text}")
        else:
            print("No response received from Watsonx model.")

    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))

if __name__ == "__main__":
    asyncio.run(main())