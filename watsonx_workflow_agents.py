import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, InstanceOf

# Enable nested event loops (useful for notebooks)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage #Import AssistantMessage for memory example
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow # Correct import for Workflow - Remove WorkflowError if not directly used
from beeai_framework.agents.factory import AgentFactoryInput, BeeAgentExecutionConfig
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

# Define State for Multi Agent Workflow - While state can be useful, Agent based workflows in BeeAI can also operate directly with Memory
# If you intend to use state, keep this, otherwise you can remove MultiAgentWorkflowState and adjust workflow instantiation
class MultiAgentWorkflowState(BaseModel):
    memory: InstanceOf[UnconstrainedMemory]

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
        memory = UnconstrainedMemory() # Initialize memory before workflow definition
        workflow = Workflow(name="MultiAgentWorkflow") #  Workflow for Agents does not require schema in constructor

        # Add WeatherForecaster agent using Watsonx
        workflow.add_step( # Correct method is add_step for workflows, agents are steps in a workflow in BeeAI framework based on provided example
            AgentFactoryInput( # AgentFactoryInput is still used within add_step to define the agent configuration as a step
                name="WeatherForecaster",
                instructions="You are a weather assistant. Respond only if you can provide a useful answer.", # Added respond only condition
                tools=[OpenMeteoTool()], # Use OpenMeteoTool for weather data
                llm=llm_watsonx,
                execution=BeeAgentExecutionConfig( # Example Execution config
                    max_iterations=3, total_max_retries=10, max_retries_per_step=3
                ),
            ), name = "WeatherForecaster" # Add name for the step, which is the agent's name for clarity
        )

        # Add Researcher agent using Watsonx
        workflow.add_step( # Correct method is add_step
            AgentFactoryInput( # AgentFactoryInput is still used
                name="Researcher",
                instructions="You are a researcher assistant. Respond only if you can provide a useful answer.", # Added respond only condition
                tools=[DuckDuckGoSearchTool()], # Use DuckDuckGoSearchTool for research
                llm=llm_watsonx,
            ), name = "Researcher" # Add name for the step, which is the agent's name for clarity
        )

        # Add Solver agent using Watsonx
        workflow.add_step( # Correct method is add_step
            AgentFactoryInput( # AgentFactoryInput is still used
                name="Solver",
                instructions=(
                    "Your task is to provide the most useful final answer based on the assistants' "
                    "responses which all are relevant. Ignore those where assistant do not know."
                ),
                llm=llm_watsonx,
            ), name = "Solver" # Add name for the step, which is the agent's name for clarity
        )


        # Create the user message
        user_message = UserMessage("What is the weather in Genova Italy?") #  Single Question for better visualization
        await memory.add(user_message) # Add user message to memory

        print("\nRunning Multi-Agent Workflow with Watsonx...")
        workflow_response = await workflow.run(memory=memory) # Run workflow with memory, passing just memory if not using state, or MultiAgentWorkflowState(memory=memory) if you intend to use state

        print("\nMulti-Agent Workflow completed.")
        print("Final Response from Workflow:\n") # Indicate final response

        # Accessing messages from memory to see agent interactions (Illustrative - might need more robust parsing for complex interactions)
        for message in workflow_response.memory.messages: # Access memory directly from workflow_response when not using state in workflow constructor
            if isinstance(message, UserMessage):
                print(f"User: {message.content}") # Print User messages
            elif isinstance(message, AssistantMessage):
                print(f"Assistant ({message.agent_name}): {message.content}") # Print Assistant messages with agent name
        final_answer = workflow_response.memory.messages[-1].content # Access memory directly from workflow_response when not using state in workflow constructor
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