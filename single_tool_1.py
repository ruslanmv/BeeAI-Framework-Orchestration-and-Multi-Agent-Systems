import asyncio
import os
import logging
from dotenv import load_dotenv

from beeai_framework.backend.chat import ChatModel
from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError

# Load environment variables (if needed)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def run_weather_tool(location: str):
    """Runs OpenMeteoTool inside a BeeAI agent workflow using Ollama."""
    logging.info(f"Initializing Ollama and Weather Agent for location: {location}")

    try:
        # Initialize Ollama Model
        llm = await ChatModel.from_name("ollama:granite3.1-dense:8b")
        logging.info("Ollama Model Loaded Successfully.")

        # Create an agent workflow with OpenMeteoTool
        workflow = AgentWorkflow(name="Weather Assistant")
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant. Provide accurate weather forecasts.",
                tools=[OpenMeteoTool()],  # Attach weather tool
                llm=llm,
                execution=BeeAgentExecutionConfig(max_iterations=3),
            )
        )

        # Store User Prompt in Memory
        memory = UnconstrainedMemory()
        await memory.add(UserMessage(content=f"What is the weather in {location}?"))

        # Run Workflow
        logging.info("Executing Weather Agent...")
        response = await workflow.run(messages=memory.messages)

        if response and response.state:
            logging.info(f"Weather Forecast for {location}: {response.state.final_answer}")
            return response.state.final_answer
        else:
            logging.warning("Weather Forecast Query Failed.")
            return None

    except WorkflowError as e:
        logging.error("WorkflowError encountered:", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return None

async def main():
    """Main function to invoke the weather tool."""
    location = "Genova, Italy"
    await run_weather_tool(location)

if __name__ == "__main__":
    asyncio.run(main())
