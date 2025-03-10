import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError

# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

async def run_workflow(prompt):
    # Initialize Watsonx model with the provided credentials.
    llm = await WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )

    try:
        workflow = AgentWorkflow(name="Smart assistant")
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant. Respond only if you can provide a useful answer.",
                tools=[OpenMeteoTool()],
                llm=llm,
                execution=BeeAgentExecutionConfig(max_iterations=3),
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Researcher",
                instructions="You are a researcher assistant. Respond only if you can provide a useful answer.",
                tools=[DuckDuckGoSearchTool()],
                llm=llm,
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Solver",
                instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
                llm=llm,
            )
        )

        # Store the initial user prompt in memory.
        memory = UnconstrainedMemory()
        await memory.add(UserMessage(content=prompt))

        # Run the workflow and return the final answer.
        response = await workflow.run(messages=memory.messages)
        return response.state.final_answer

    except WorkflowError:
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return None

def execute_in_normal_python(prompt):
    """Executes the workflow in a normal Python environment."""
    try:
        result = asyncio.run(run_workflow(prompt))
        if result:
            print(f"result: {result}")
        else:
            print("Workflow execution failed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

prompt = "What is the weather in Genova Italy?"
execute_in_normal_python(prompt)
