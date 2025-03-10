import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Enable nested event loops (useful for notebooks)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage #Import AssistantMessage for memory example
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowError
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow,BeeAgentExecutionConfig
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool 
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool 
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory 


# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")


async def evaluate_model(llm, query: str) -> str:
    """
    Evaluates a given language model with a query and returns the text response.
    """
    user_message = UserMessage(content=query)
    try:
        response = await asyncio.wait_for(llm.create({"messages": [user_message]}), timeout=30)
        return response.get_text_content()
    except Exception as e:
        print(f"Error evaluating model {llm.__class__.__name__}: {e}")
        return ""


async def main() -> None:
    """Main function to demonstrate BeeAI Workflow with multiple Watsonx-powered agents."""
    print("Starting BeeAI Workflow with Multiple Watsonx Agents Example...\n")

    try:
        print("Initializing Watsonx model for agents...")
        llm_watsonx = await WatsonxChatModel.from_name( # Initialize Watsonx model globally for agents
            "watsonx:ibm/granite-3-8b-instruct",
            options={
                "project_id": WATSONX_PROJECT_ID,
                "api_key": WATSONX_API_KEY,
                "api_base": WATSONX_API_URL,
            },
        )
        print("Watsonx model initialized for agents.")

        # --- Create Workflow with Agents ---
        workflow = Workflow(name="MultiAgentWorkflow") # No schema needed for agent-based workflows as per doc example

        # Add WeatherForecaster agent using Watsonx
        workflow.add_agent(
            AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant. Use the open meteo tool to get weather information.",
                tools=[OpenMeteoTool()], # Use OpenMeteoTool for weather data
                llm=llm_watsonx,
                execution=BeeAgentExecutionConfig( # Example Execution config
                    max_iterations=3, total_max_retries=10, max_retries_per_step=3
                ),
            )
        )

        # Add Researcher agent using Watsonx
        workflow.add_agent(
            AgentFactoryInput(
                name="Researcher",
                instructions="You are a research assistant. Use the duck duck go search tool to find information.",
                tools=[DuckDuckGoSearchTool()], # Use DuckDuckGoSearchTool for research
                llm=llm_watsonx,
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

        memory = UnconstrainedMemory() # Initialize memory for the workflow
        # Create the user message
        user_message = UserMessage("What is the weather in London and the capital of France?")
        await memory.add(user_message) # Add user message to memory

        print("\nRunning Multi-Agent Workflow with Watsonx...")
        workflow_response = await workflow.run(memory=memory) # Run workflow with memory

        print("\nMulti-Agent Workflow completed.")
        print("Final Response from Workflow:\n") # Indicate final response

        # Accessing messages from memory to see agent interactions (Illustrative - might need more robust parsing for complex interactions)
        for message in workflow_response.memory.messages: # Iterate through memory messages
            if isinstance(message, UserMessage):
                print(f"User: {message.content}") # Print User messages
            elif isinstance(message, AssistantMessage):
                print(f"Assistant ({message.agent_name}): {message.content}") # Print Assistant messages with agent name
        final_answer = workflow_response.memory.messages[-1].content # Assume last message in memory is the final answer
        print(f"\nSolver's Final Answer: {final_answer}") # Print Solver's assumed final answer


    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
    except ValidationError:
        traceback.print_exc()
        sys.exit(str(e))
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))

    print("\nBeeAI Workflow with Multiple Watsonx Agents Example finished.")


if __name__ == "__main__":
    asyncio.run(main())