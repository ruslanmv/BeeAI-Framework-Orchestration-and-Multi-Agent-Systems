import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv

# Optional: Apply nest_asyncio to allow asyncio.run() in environments with a running loop (e.g., Jupyter)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from beeai_framework.agents.types import BeeAgentExecutionConfig
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

async def test_model(llm, query: str) -> str:
    """
    Test a given language model by sending a query and returning the response text.
    """
    user_message = UserMessage(content=query)
    try:
        response = await llm.create({"messages": [user_message]})
        result_text = response.get_text_content()
        print(f"Response from model {llm.__class__.__name__}: {result_text}")
        return result_text
    except Exception as e:
        print(f"Error testing model {llm.__class__.__name__}: {e}")
        return ""

async def run_mixed_workflow() -> None:
    """
    Set up and run the mixed-backend workflow using Watsonx and Ollama.
    """
    # Initialize Watsonx ChatModel using environment variables
    llm_watsonx = await WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    # Initialize Ollama ChatModel (no env options required)
    llm_ollama = OllamaChatModel("granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Smart assistant (Mixed Backend)")
    
    # Add WeatherForecaster agent using Watsonx
    workflow.add_agent(
        AgentFactoryInput(
            name="WeatherForecaster",
            instructions="You are a weather assistant.",
            tools=[OpenMeteoTool()],
            llm=llm_watsonx,
            execution=BeeAgentExecutionConfig(
                max_iterations=3, total_max_retries=10, max_retries_per_step=3
            ),
        )
    )
    
    # Add Researcher agent using Ollama
    workflow.add_agent(
        AgentFactoryInput(
            name="Researcher",
            instructions="You are a researcher assistant.",
            tools=[DuckDuckGoSearchTool()],
            llm=llm_ollama,
        )
    )
    
    # Add Solver agent using Watsonx
    workflow.add_agent(
        AgentFactoryInput(
            name="Solver",
            instructions=(
                "Your task is to provide the most useful final answer based on the assistants' "
                "responses which all are relevant. Ignore those where assistant do not know."
            ),
            llm=llm_watsonx,
        )
    )

    prompt = "What is the weather in London and the capital of France?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    
    response = await workflow.run(messages=memory.messages)
    print(f"Result from mixed workflow: {response.state.final_answer}")

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
    llm_ollama = OllamaChatModel("granite3.1-dense:8b")
    await test_model(llm_ollama, "What is the capital of Italy?")

    # Run the mixed backend workflow
    print("\nRunning mixed backend workflow:")
    await run_mixed_workflow()

if __name__ == "__main__":
    try:
        # This allows the code to run in environments (like Jupyter) with an already running event loop.
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))
