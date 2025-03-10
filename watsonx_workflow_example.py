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
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowError

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

# 2. Watsonx Simple Question Answering Workflow (Focused Example)
class QuestionState(BaseModel):
    question: str
    answer: str | None = None

async def answer_question_step(state: QuestionState) -> str:
    """Workflow step to answer a question using Watsonx."""
    print("Running workflow step: answer_question_step") # Indicate step execution
    llm_watsonx = await WatsonxChatModel.from_name( # Initialize Watsonx model within the step
        "watsonx:ibm/granite-3-8b-instruct",
        options={
            "project_id": WATSONX_PROJECT_ID,
            "api_key": WATSONX_API_KEY,
            "api_base": WATSONX_API_URL,
        },
    )
    query = state.question
    print(f"Sending query to Watsonx: '{query}'") # Indicate query sending
    response_text = await evaluate_model(llm_watsonx, query)
    state.answer = response_text
    print("Watsonx answered. Step complete.") # Indicate step completion
    return Workflow.END


async def main() -> None:
    """Main function to demonstrate BeeAI Watsonx Question Answering Workflow."""
    print("Starting BeeAI Watsonx Question Answering Workflow Example...\n") # Clear start message

    # --- 2. Run Watsonx Question Answering Workflow ---
    try:
        question_workflow = Workflow(schema=QuestionState, name="WatsonxQuestionWorkflow")
        question_workflow.add_step("answer_question_step", answer_question_step)
        # Define a single, clear question for demonstration
        demonstration_question = "What is the capital of France?"
        question_response = await question_workflow.run(QuestionState(question=demonstration_question))

        print("\nWatsonx Question Answering Workflow completed.") # Indicate workflow completion
        print("Question asked:", question_response.state.question) # Show the question
        print("Answer from Watsonx:", question_response.state.answer) # Show the answer

    except WorkflowError as e:
        traceback.print_exc()
        sys.exit(e.explain())
    except ValidationError:
        traceback.print_exc()
        sys.exit(str(e))
    except Exception as e:
        traceback.print_exc()
        sys.exit(str(e))

    print("\nBeeAI Watsonx Question Answering Workflow Example finished.") # Clear finish message


if __name__ == "__main__":
    asyncio.run(main())