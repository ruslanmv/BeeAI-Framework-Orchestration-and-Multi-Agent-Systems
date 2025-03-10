import asyncio
import sys
import traceback
import os
import json # Import json for tool arguments handling
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Enable nested event loops (useful in notebooks)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage, ToolMessage, MessageToolResultContent # Import ToolMessage and related classes
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool # Import OpenMeteoTool

# Load environment variables from .env
load_dotenv()
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_API_URL")


async def evaluate_model(llm, query: str, tools=None) -> str: # Add tools argument
    """
    Evaluate a given Watsonx model with a query and return the text response.
    """
    print("DEBUG: Evaluating model with query:", query, "with tools:", tools is not None) # Debug tools info
    user_message = UserMessage(content=query)
    try:
        if tools: # Conditionally use tools in create call
            response = await asyncio.wait_for(llm.create({"messages": [user_message], "tools": tools}), timeout=30)
        else:
            response = await asyncio.wait_for(llm.create({"messages": [user_message]}), timeout=30)
        return response.get_text_content()
    except Exception as e:
        print(f"DEBUG: Error evaluating model {llm.__class__.__name__}: {e}")
        return ""


# 1. Basic Workflow Example
class MessageState(BaseModel):
    message: str


async def my_first_step(state: MessageState) -> str:
    """
    A simple workflow step that appends text to the message.
    """
    print("DEBUG: Running my_first_step with initial message:", state.message)
    state.message += " from Watsonx!"
    print("DEBUG: Modified message:", state.message)
    return Workflow.END


# 2. Watsonx Question Answering Workflow
class QuestionState(BaseModel):
    question: str
    answer: str | None = None


async def answer_question_step(state: QuestionState) -> str:
    """
    Workflow step that initializes the Watsonx model, sends a question, and saves the answer.
    """
    print("DEBUG: Running answer_question_step with question:", state.question)
    llm_watsonx = await WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    print("DEBUG: Watsonx model initialized for question answering.")
    response_text = await evaluate_model(llm_watsonx, state.question)
    state.answer = response_text
    print("DEBUG: Watsonx answered with:", response_text)
    return Workflow.END


# 3. Watsonx Agent Workflow with Tool Calling
class AgentQuestionState(BaseModel):
    question: str
    answer: str | None = None

async def agent_answer_question_step(state: AgentQuestionState) -> str:
    """
    Workflow step that uses Watsonx with tool calling to answer a question, potentially using tools.
    """
    print("DEBUG: Running agent_answer_question_step with question:", state.question)
    llm_watsonx = await WatsonxChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    print("DEBUG: Watsonx model initialized for agent with tool calling.")
    weather_tool = OpenMeteoTool() # Initialize the OpenMeteoTool

    user_message = UserMessage(content=state.question)
    response = await llm_watsonx.create(messages=[user_message], tools=[weather_tool]) # Send message and tools

    if response.has_tool_calls(): # Check if the response contains tool calls
        print("DEBUG: Tool call detected!")
        tool_call_msg = response.get_tool_calls()[0] # Assume single tool call for simplicity
        print("DEBUG: Tool call details:", tool_call_msg.model_dump())
        tool_response = await weather_tool.run(json.loads(tool_call_msg.args)) # Execute the tool
        tool_response_msg = ToolMessage(
            MessageToolResultContent(
                result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
            )
        )
        print("DEBUG: Tool response:", tool_response_msg.to_plain())
        final_response = await llm_watsonx.create(messages=[user_message, tool_response_msg], tools=[]) # Send tool response back to LLM
        state.answer = final_response.get_text_content()
        print("DEBUG: Agent answered with tool:", state.answer)

    else: # No tool call, just get text content as before
        response_text = response.get_text_content()
        state.answer = response_text
        print("DEBUG: Watsonx agent answered without tool:", response_text)

    return Workflow.END


async def main() -> None:
    print("DEBUG: Starting BeeAI Workflow Example with Watsonx...\n")

    # Run Basic Workflow Example
    print("--- Running Basic Workflow Example ---")
    try:
        basic_workflow = Workflow(schema=MessageState, name="BasicWorkflowExample")
        basic_workflow.add_step("my_first_step", my_first_step)
        basic_response = await basic_workflow.run(MessageState(message="Hello"))
        print("DEBUG: Basic Workflow completed.")
        print("DEBUG: Final Message State:", basic_response.state)
    except (WorkflowError, ValidationError) as e:
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))

    # Run Watsonx Question Answering Workflow
    print("\n--- Running Watsonx Question Answering Workflow ---")
    try:
        question_workflow = Workflow(schema=QuestionState, name="WatsonxQuestionWorkflow")
        question_workflow.add_step("answer_question_step", answer_question_step)
        question_response = await question_workflow.run(
            QuestionState(question="What is the highest mountain in the world?")
        )
        print("DEBUG: Watsonx Question Answering Workflow completed.")
        print("DEBUG: Question:", question_response.state.question)
        print("DEBUG: Answer from Watsonx:", question_response.state.answer)
    except (WorkflowError, ValidationError) as e:
        traceback.print_exc()
        sys.exit(e.explain() if isinstance(e, FrameworkError) else str(e))
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))

    # Run Watsonx Agent Workflow with Tool Calling
    print("\n--- Running Watsonx Agent Workflow with Tool Calling ---")
    try:
        agent_workflow = Workflow(schema=AgentQuestionState, name="WatsonxAgentWorkflow")
        agent_workflow.add_step("agent_answer_question_step", agent_answer_question_step)
        agent_response = await agent_workflow.run(
            AgentQuestionState(question="What is the current weather in London?") # Example question for weather tool
        )
        print("DEBUG: Watsonx Agent Workflow with Tool Calling completed.")
        print("DEBUG: Question:", agent_response.state.question)
        print("DEBUG: Answer from Watsonx Agent:", agent_response.state.answer)
    except (WorkflowError, ValidationError) as e:
        traceback.print_exc()
        sys.exit(e.explain() if isinstance(e, FrameworkError) else str(e))
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))


    print("\nDEBUG: BeeAI Workflow Example with Watsonx finished.")


if __name__ == "__main__":
    asyncio.run(main())