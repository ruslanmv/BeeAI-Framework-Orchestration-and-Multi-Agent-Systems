import asyncio
import traceback
import os
import json
import sys
from dotenv import load_dotenv
from pydantic import ValidationError, BaseModel, Field

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel, ChatModelStructureInput # Import ChatModelStructureInput
from beeai_framework.backend.message import UserMessage, ToolResult
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework import ToolMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError, FrameworkError


load_dotenv()

WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_URL")

llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    settings={
        "project_id": WATSONX_PROJECT_ID,
        "api_key": WATSONX_API_KEY,
        "api_base": WATSONX_API_URL,
    },
)


async def watsonx_from_name() -> None:
    watsonx_llm = await ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        {
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    user_message = UserMessage("what states are part of New England?")
    chat_model_input = {
        "messages": [user_message]
    }
    response = await watsonx_llm.create(chat_model_input)
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    chat_model_input = {
        "messages": [user_message]
    }
    response = await llm.create(chat_model_input)
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    chat_model_input = {
        "messages": [user_message],
        "stream": True # Added stream to chat_model_input
    }
    response = await llm.create(chat_model_input) # Removed stream=True here
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        chat_model_input = {
            "messages": [user_message],
            "abort_signal": AbortSignal.timeout(0.5), # Added abort_signal to chat_model_input
            "stream": True # Added stream to chat_model_input
        }
        response = await llm.create(chat_model_input) # Removed stream=True, abort_signal=AbortSignal.timeout(0.5) (from here)

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def watson_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    chat_model_input = ChatModelStructureInput( # Create ChatModelStructureInput instance
        messages=[user_message],
        schema=TestSchema
    )
    response = await llm.create_structure(chat_model_input) # Pass ChatModelStructureInput instance as positional argument
    print(response.object)


async def watson_tool_calling() -> None:
    watsonx_llm = await ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        {
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    user_message = UserMessage("What is the current weather in Boston?")
    weather_tool = OpenMeteoTool()
    chat_model_input = {
        "messages": [user_message],
        "tools": [weather_tool] # Added tools to chat_model_input
    }
    response = await watsonx_llm.create(chat_model_input) # Removed tools=[weather_tool] from here
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        ToolResult(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response_input = {
        "messages": [user_message, tool_response_msg]
    }
    final_response = await watsonx_llm.create(final_response_input, tools=[]) # Keep tools=[] here for final response, as per original code
    print(final_response.get_text_content())


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()
    print("*" * 10, "watson_tool_calling")
    await watson_tool_calling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())