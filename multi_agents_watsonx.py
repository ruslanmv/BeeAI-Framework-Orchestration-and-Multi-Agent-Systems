import asyncio
import os
import sys
import traceback
import logging
from dotenv import load_dotenv

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError

# Load environment variables
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def test_model(llm, query: str) -> str:
    """Send a query to the Watsonx language model and return its response."""
    logging.info(f"Sending query to Watsonx: '{query}'")

    # FIX: Convert message into the required Pydantic model
    user_message = UserMessage(content=query)

    try:
        # Ensure the correct format by wrapping messages in a list
        response = await asyncio.wait_for(llm.create({"messages": [user_message]}), timeout=30)
        result_text = response.get_text_content()
        logging.info(f"Watsonx Response: {result_text}")
        return result_text
    except Exception as e:
        logging.error(f"Error testing Watsonx model: {e}", exc_info=True)
        return ""

async def main() -> None:
    """Main function to run the Watsonx model test."""
    logging.info("Initializing Watsonx Model...")

    try:
        # Initialize Watsonx Model
        llm_watsonx = await WatsonxChatModel.from_name(
            "watsonx:ibm/granite-3-8b-instruct",
            options={
                "project_id": WATSONX_PROJECT_ID,
                "api_key": WATSONX_API_KEY,
                "api_base": WATSONX_API_URL,
            },
        )
        logging.info("Watsonx Model Loaded Successfully.")

        # Run Weather Forecast Query
        prompt = "What is the weather in Genova, Italy?"
        result = await test_model(llm_watsonx, prompt)

        if result:
            logging.info(f"Final Result: {result}")
        else:
            logging.warning("Weather Forecast Query Failed.")

    except FrameworkError as e:
        logging.error("FrameworkError encountered:", exc_info=True)
        sys.exit(e.explain())
    except Exception as e:
        logging.error("Unexpected error encountered:", exc_info=True)
        sys.exit(str(e))

if __name__ == "__main__":
    asyncio.run(main())
