# BeeAI Framework: Your Guide from Zero to Hero

Welcome to the BeeAI Framework! This repository is your comprehensive guide to learning and using BeeAI, taking you from an absolute beginner to a proficient developer. In this extended tutorial we cover:

- **Prompt Templates:** How to create and render templates for dynamic prompt generation.
- **ChatModel Interaction:** How to interact with a language model using message-based inputs.
- **Memory Handling:** How to build a conversation history for context.
- **Structured Outputs:** Enforcing output formats with Pydantic schemas.
- **System Prompts:** Guiding LLM behavior with system messages.
- **ReAct Agents and Tools:** Building agents that can reason and act, including integration with external tools.
- **Workflows:** Combining all of the above into a multi-step process, including adding memory and orchestration of multi-agent systems.

Below you’ll find all the code examples along with explanations.

---

## BeeAI Framework Basics

These examples demonstrate the fundamental usage patterns of BeeAI in Python. They progressively increase in complexity, providing a well-rounded overview of the framework.

---

### 1. Prompt Templates

One of the core constructs in the BeeAI framework is the `PromptTemplate`. It allows you to dynamically insert data into a prompt before sending it to a language model. BeeAI uses the Mustache templating language for prompt formatting.

#### Example: RAG Prompt Template

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Define the structure of the input data that will be passed to the template.
class RAGTemplateInput(BaseModel):
    question: str
    context: str

# Define the prompt template.
rag_template: PromptTemplate = PromptTemplate(
    schema=RAGTemplateInput,
    template="""
Context: {{context}}
Question: {{question}}

Provide a concise answer based on the context. Avoid statements such as 'Based on the context' or 'According to the context' etc. """,
)

# Render the template using an instance of the input model.
prompt = rag_template.render(
    RAGTemplateInput(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
    )
)

# Print the rendered prompt.
print(prompt)
```

---

### 2. More Complex Templates

The `PromptTemplate` class also supports more complex structures. For example, you can iterate over a list of search results to build a prompt.

#### Example: Template with a List of Search Results

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Individual search result schema.
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

# Input specification for the template.
class SearchTemplateInput(BaseModel):
    question: str
    results: list[SearchResult]

# Define the template that iterates over the search results.
search_template: PromptTemplate = PromptTemplate(
    schema=SearchTemplateInput,
    template="""
Search results:
{{#results.0}}
{{#results}}
Title: {{title}}
Url: {{url}}
Content: {{content}}
{{/results}}
{{/results.0}}

Question: {{question}}
Provide a concise answer based on the search results provided.""",
)

# Render the template with sample data.
prompt = search_template.render(
    SearchTemplateInput(
        question="What is the capital of France?",
        results=[
            SearchResult(
                title="France",
                url="[https://en.wikipedia.org/wiki/France](https://en.wikipedia.org/wiki/France)",
                content="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
            )
        ],
    )
)

# Print the rendered prompt.
print(prompt)
```

---

### 3. The ChatModel

Once you have your prompt templates set up, you can begin interacting with a language model. BeeAI supports various LLMs through the `ChatModel` interface.

#### Example: Creating a User Message

```python
from beeai_framework.backend.message import UserMessage

# Create a user message to start a chat with the model.
user_message = UserMessage(content="Hello! Can you tell me what is the capital of France?")
```

#### Example: Sending a Message to the ChatModel

```python
from beeai_framework.backend.chat import ChatModel, ChatModelInput, ChatModelOutput

# Create a ChatModel instance that interfaces with Granite 3.1 (via Ollama).
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Send the user message and get the model's response.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[user_message]))

# Print the model's response.
print(output.get_text_content())
```

---

### 4. Memory Handling

Memory is a convenient way to store the conversation history (a series of messages) that the model uses for context.

#### Example: Storing and Retrieving Conversation History

```python
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create an unconstrained memory instance.
memory = UnconstrainedMemory()

# Add a series of messages to the memory.
await memory.add_many(
    [
        user_message,
        AssistantMessage(content=output.get_text_content()),
        UserMessage(content="If you had to recommend one thing to do there, what would it be?"),
    ]
)

# Send the complete message history to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=memory.messages))
print(output.get_text_content())
```

---

### 5. Combining Templates and Messages

You can render a prompt from a template and then send it as a message to the ChatModel.

#### Example: Rendering a Template and Sending as a Message

```python
# Some context that the model will use (e.g., from Wikipedia on Ireland).
context = """The geography of Ireland comprises relatively low-lying mountains surrounding a central plain, with several navigable rivers extending inland.
Its lush vegetation is a product of its mild but changeable climate which is free of extremes in temperature.
Much of Ireland was woodland until the end of the Middle Ages. Today, woodland makes up about 10% of the island,
compared with a European average of over 33%, with most of it being non-native conifer plantations.
The Irish climate is influenced by the Atlantic Ocean and thus very moderate, and winters are milder than expected for such a northerly area,
although summers are cooler than those in continental Europe. Rainfall and cloud cover are abundant.
"""

# Reuse the previously defined RAG template.
prompt = rag_template.render(RAGTemplateInput(question="How much of Ireland is forested?", context=context))

# Send the rendered prompt to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[UserMessage(content=prompt)]))
print(output.get_text_content())
```

---

### 6. Structured Outputs

Sometimes you want the LLM to generate output in a specific format. You can enforce this using structured outputs with a Pydantic schema.

#### Example: Enforcing a Specific Output Format

```python
from typing import Literal
from pydantic import Field
from beeai_framework.backend.chat import ChatModelStructureInput

# Define the output structure for a character.
class CharacterSchema(BaseModel):
    name: str = Field(description="The name of the character.")
    occupation: str = Field(description="The occupation of the character.")
    species: Literal["Human", "Insectoid", "Void-Serpent", "Synth", "Ethereal", "Liquid-Metal"] = Field(
        description="The race of the character."
    )
    back_story: str = Field(description="Brief backstory of this character.")

# Create a user message instructing the model to generate a character.
user_message = UserMessage(
    "Create a fantasy sci-fi character for my new game. This character will be the main protagonist, be creative."
)

# Request a structured response from the model.
response = await model.create_structure(ChatModelStructureInput(schema=CharacterSchema, messages=[user_message]))
print(response.object)
```

---

### 7. System Prompts

System messages can guide the overall behavior of the language model.

#### Example: Using a System Message

```python
from beeai_framework.backend.message import SystemMessage

# Create a system message that instructs the LLM to respond like a pirate.
system_message = SystemMessage(content="You are pirate. You always respond using pirate slang.")

# Create a new user message.
user_message = UserMessage(content="What is a baby hedgehog called?")

# Send both messages to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[system_message, user_message]))
print(output.get_text_content())
```

---

## BeeAI ReAct Agents

The BeeAI ReAct agent implements the “Reasoning and Acting” pattern, separating the process into distinct steps. This section shows how to build an agent that uses its own memory for reasoning and even integrates tools for added functionality.

### 1. Basic ReAct Agent

#### Example: Setting Up a Basic ReAct Agent

```python
from typing import Any
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Construct the BeeAgent without external tools.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[], memory=UnconstrainedMemory()))

# Define a function to process agent events.
async def process_agent_events(event_data: dict[str, Any], event_meta: EventMeta) -> None:
    if event_meta.name == "error":
        print("Agent 🤖 : ", event_data["error"])
    elif event_meta.name == "retry":
        print("Agent 🤖 : ", "retrying the action...")
    elif event_meta.name == "update":
        print(f"Agent({event_data['update']['key']}) 🤖 : ", event_data["update"]["parsedValue"])

# Attach an observer to log events.
async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_agent_events, EmitterOptions(match_nested=True))

# Run the agent with a sample prompt.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What chemical elements make up a water molecule?")
).observe(observer)
```

---

### 2. Using Tools with the Agent

Agents can be extended with tools so that they can perform external actions, like fetching weather data.

#### Example: Using a Built-In Weather Tool

```python
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that includes the OpenMeteoTool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

# Run the agent with a prompt about the weather.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What's the current weather in London?")
).observe(observer)
```

---

### 3. Imported Tools

You can also import tools from other libraries. Below are two examples that show how to integrate Wikipedia search via LangChain.

#### Example: Long-Form Integration with Wikipedia

```python
from typing import Any
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool
from beeai_framework.tools.tool import StringToolOutput

# Define the input schema for the Wikipedia tool.
class LangChainWikipediaToolInput(BaseModel):
    query: str = Field(description="The topic or question to search for on Wikipedia.")

# Adapter class to integrate LangChain's Wikipedia tool.
class LangChainWikipediaTool(Tool):
    name = "Wikipedia"
    description = "Search factual and historical information from Wikipedia about given topics."
    input_schema = LangChainWikipediaToolInput

    def __init__(self) -> None:
        super().__init__()
        wikipedia = WikipediaAPIWrapper()
        self.wikipedia = WikipediaQueryRun(api_wrapper=wikipedia)

    def _run(self, input: LangChainWikipediaToolInput, _: Any | None = None) -> None:
        query = input.query
        try:
            result = self.wikipedia.run(query)
            return StringToolOutput(result=result)
        except Exception as e:
            print(f"Wikipedia search error: {e!s}")
            return f"Error searching Wikipedia: {e!s}"

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the custom Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[LangChainWikipediaTool()], memory=UnconstrainedMemory()))

# Run the agent with a query about the European Commission.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="Who is the current president of the European Commission?")
).observe(observer)
```

#### Example: Shorter Form Using the `@tool` Decorator

```python
from langchain_community.tools import WikipediaQueryRun  # noqa: F811
from langchain_community.utilities import WikipediaAPIWrapper  # noqa: F811
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool, tool

# Define a tool using the decorator.
@tool
def langchain_wikipedia_tool(expression: str) -> str:
    """
    Search factual and historical information, including biography, history, politics, geography, society, culture,
    science, technology, people, animal species, mathematics, and other subjects.
    
    Args:
        expression: The topic or question to search for on Wikipedia.
    
    Returns:
        The information found via searching Wikipedia.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return StringToolOutput(wikipedia.run(expression))

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the decorated Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[langchain_wikipedia_tool], memory=UnconstrainedMemory()))

# Run the agent with a query about the longest living vertebrate.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What is the longest living vertebrate?")
).observe(observer)
```

---

## BeeAI Workflows (Experimental)

Workflows allow you to combine what you’ve learned into a coherent multi-step process. A workflow is defined by a state (a Pydantic model) and steps (Python functions) that update the state and determine the next step. Workflows in BeeAI provide a flexible and extensible component for managing and executing structured sequences of tasks, especially useful for orchestration of complex agent behaviors and multi-agent systems.

### Overview

Workflows provide a flexible and extensible component for managing and executing structured sequences of tasks. They are particularly useful for:

- 🔄 **Dynamic Execution**: Steps can direct the flow based on state or results
- ✅ **Validation**: Define schemas for data consistency and type safety
- 🧩 **Modularity**: Steps can be standalone or invoke nested workflows
- 👁️ **Observability**: Emit events during execution to track progress or handle errors

---

### Core Concepts

#### State

State is the central data structure in a workflow. It's a Pydantic model that:
- Holds the data passed between steps
- Provides type validation and safety
- Persists throughout the workflow execution

#### Steps

Steps are the building blocks of a workflow. Each step is a function that:
- Takes the current state as input
- Can modify the state
- Returns the name of the next step to execute or a special reserved value

#### Transitions

Transitions determine the flow of execution between steps. Each step returns either:
- The name of the next step to execute
- `Workflow.NEXT` - proceed to the next step in order
- `Workflow.SELF` - repeat the current step
- `Workflow.END` - end the workflow execution

---

### Basic Usage

#### Simple Workflow

The example below demonstrates a minimal workflow that processes steps in sequence. This pattern is useful for straightforward, linear processes where each step builds on the previous one.

```python
import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow


async def main() -> None:
    # State
    class State(BaseModel):
        input: str

    workflow = Workflow(State)
    workflow.add_step("first", lambda state: print("Running first step!"))
    workflow.add_step("second", lambda state: print("Running second step!"))
    workflow.add_step("third", lambda state: print("Running third step!"))

    await workflow.run(State(input="Hello"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

#### Multi-Step Workflow

This advanced example showcases a workflow that implements multiplication through repeated addition—demonstrating control flow, state manipulation, nesting, and conditional logic.

```python
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowReservedStepName

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This workflow demonstrates several powerful concepts:
- Implementing loops by returning `Workflow.SELF`
- Conditional transitions between steps
- Progressive state modification to accumulate results
- Sign handling through state transformation
- Type-safe step transitions using Literal types

---

### Advanced Features

#### Workflow Nesting

Workflow nesting allows complex behaviors to be encapsulated as reusable components, enabling hierarchical composition of workflows. This promotes modularity, reusability, and better organization of complex agent logic.

```python
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow, WorkflowReservedStepName

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

#### Multi-Agent Workflows: Orchestration with BeeAI

The multi-agent workflow pattern enables the orchestration of specialized agents that collaborate to solve complex problems. Each agent focuses on a specific domain or capability, with results combined by a coordinator agent.  BeeAI Framework's workflow engine is perfectly suited for creating sophisticated multi-agent systems.

The following example demonstrates how to orchestrate a multi-agent system using BeeAI workflows with Ollama backend. We will create a "Smart assistant" workflow composed of three specialized agents: `WeatherForecaster`, `Researcher`, and `Solver`.

```python
import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Smart assistant")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant. Use tools to provide weather information.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant. Use search tools to find information.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the weather in New York?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result (Ollama Backend): {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This pattern demonstrates:

- **Role specialization** through focused agent configuration. `WeatherForecaster` is designed specifically for weather-related queries, while `Researcher` is for general information retrieval.
- **Efficient tool distribution** to relevant specialists. The `WeatherForecaster` agent is equipped with the `OpenMeteoTool`, and `Researcher` with `DuckDuckGoSearchTool`, ensuring each agent has the right tools for its job.
- **Parallel processing** of different aspects of a query.  Although not explicitly parallel in this example, the workflow structure is designed to easily support parallel execution of agents if needed.
- **Synthesis of multiple expert perspectives** into a cohesive response. The `Solver` agent acts as a coordinator, taking responses from other agents and synthesizing them into a final answer.
- **Declarative agent configuration** using the `AgentWorkflow` and `add_agent` methods, which simplifies the setup and management of complex agent systems.

**Orchestration with Watsonx.ai Backend**

To demonstrate the versatility of BeeAI workflows, let's adapt the multi-agent workflow example to use Watsonx.ai as the backend LLM provider. First, ensure you have configured the Watsonx provider as described in the Backend section. Then, modify the `ChatModel.from_name` call to use a Watsonx model:

```python
import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow

async def main() -> None:
    # Initialize Watsonx ChatModel
    llm = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct") # Replace with your Watsonx model

    workflow = AgentWorkflow(name="Smart assistant (Watsonx)")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the weather in London?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result (Watsonx Backend): {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```
In this modified example, we simply changed the `ChatModel.from_name` call to `watsonx:ibm/granite-3-8b-instruct`.  Assuming you have correctly set up your Watsonx environment variables, this code will now orchestrate the same multi-agent workflow but powered by Watsonx.ai. This highlights the provider-agnostic nature of BeeAI workflows, allowing you to easily switch between different LLM backends without significant code changes.

#### Memory in Workflows

Integrating memory into workflows allows agents to maintain context across interactions, enabling conversational interfaces and stateful processing. This example demonstrates a simple conversational echo workflow with persistent memory.

```python
import asyncio
import sys
import traceback

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows.workflow import Workflow
from examples.helpers.io import ConsoleReader


async def main() -> None:
    # State with memory
    class State(BaseModel):
        memory: InstanceOf[UnconstrainedMemory]
        output: str | None = None

    async def echo(state: State) -> str:
        # Get the last message in memory
        last_message = state.memory.messages[-1]
        state.output = last_message.text[::-1]
        return Workflow.END

    reader = ConsoleReader()

    memory = UnconstrainedMemory()
    workflow = Workflow(State)
    workflow.add_step("echo", echo)

    for prompt in reader:
        # Add user message to memory
        await memory.add(UserMessage(content=prompt))
        # Run workflow with memory
        response = await workflow.run(State(memory=memory))
        # Add assistant response to memory
        await memory.add(AssistantMessage(content=response.state.output))

        reader.write("Assistant 🤖 : ", response.state.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

This pattern demonstrates:
- Integration of memory as a first-class citizen in workflow state
- Conversation loops that preserve context across interactions
- Bidirectional memory updating (reading recent messages, storing responses)
- Clean separation between the persistent memory and workflow-specific state

---

## ⚙️ Backend

## Table of Contents
- [Overview](#overview)
- [Supported Providers](#supported-providers)
- [Backend Initialization](#backend-initialization)
- [Chat Model](#chat-model)
  - [Chat Model Configuration](#chat-model-configuration)
  - [Text Generation](#text-generation)
  - [Streaming Responses](#streaming-responses)
  - [Structured Generation](#structured-generation)
  - [Tool Calling](#tool-calling)
- [Embedding Model](#embedding-model)
  - [Embedding Model Configuration](#embedding-model-configuration)
  - [Embedding Model Usage](#embedding-model-usage)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
---

## Overview

Backend is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (ChatModel class)
- Embedding Models (coming soon)
- Audio Models (coming soon)
- Image Models (coming soon)

BeeAI framework's backend is designed with a provider-based architecture, allowing you to switch between different AI service providers while maintaining a consistent API.

> [!NOTE]
>
> Location within the framework: [beeai_framework/backend](/python/beeai_framework/backend).

---

## Supported providers

The following table depicts supported providers. Each provider requires specific configuration through environment variables. Ensure all required variables are set before initializing a provider.

| Name             | Chat | Embedding | Dependency               | Environment Variables                                                                                                                                                 |
| ---------------- | :--: | :-------: | ------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ollama         |  ✅  |          | ollama-ai-provider     | OLLAMA_CHAT_MODEL<br/>OLLAMA_BASE_URL                                                                                                       |
| OpenAI         |  ✅  |          | openai     | OPENAI_CHAT_MODEL<br/>OPENAI_API_BASE<br/>OPENAI_API_KEY<br/>OPENAI_ORGANIZATION                                                                                                       |
| Watsonx        |  ✅  |          | @ibm-cloud/watsonx-ai  | WATSONX_CHAT_MODEL<br/>WATSONX_EMBEDDING_MODEL<br>WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_SPACE_ID<br>WATSONX_VERSION<br>WATSONX_REGION                    |
| Groq           |  ✅  |         | | GROQ_CHAT_MODEL<br>GROQ_API_KEY |
| Amazon Bedrock |  ✅  |         |  boto3| AWS_CHAT_MODEL<br>AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY<br>AWS_REGION_NAME |
| Google Vertex  |  ✅  |         |  | VERTEXAI_CHAT_MODEL<br>VERTEXAI_PROJECT<br>GOOGLE_APPLICATION_CREDENTIALS<br>GOOGLE_APPLICATION_CREDENTIALS_JSON<br>GOOGLE_CREDENTIALS |
| Azure OpenAI   |    |         | Coming soon! | AZURE_OPENAI_CHAT_MODEL<br>AZURE_OPENAI_EMBEDDING_MODEL<br>AZURE_OPENAI_API_KEY<br>AZURE_OPENAI_API_ENDPOINT<br>AZURE_OPENAI_API_RESOURCE<br>AZURE_OPENAI_API_VERSION |
| Anthropic      |  ✅  |         |  | ANTHROPIC_CHAT_MODEL<br>ANTHROPIC_API_KEY |
| xAI           |  ✅  |         | | XAI_CHAT_MODEL<br>XAI_API_KEY |


> [!TIP]
>
> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions). Meanwhile, you can use [Ollama adapter](/python/examples/backend/providers/ollama.py).

---

### Backend initialization

The Backend class serves as a central entry point to access models from your chosen provider.

**Watsonx Initialization**

To use Watsonx with BeeAI framework, you need to install the Watsonx adapter and set up your environment variables.

**Installation:**

```bash
pip install beeai-framework[watsonx]
```

**Environment Variables:**

Set the following environment variables. You can obtain these from your IBM Cloud account and Watsonx service instance.

- `WATSONX_API_KEY`: Your Watsonx API key.
- `WATSONX_PROJECT_ID`: Your Watsonx project ID.
- `WATSONX_REGION`: The region where your Watsonx service is deployed (e.g., `us-south`).
- `WATSONX_CHAT_MODEL`: The specific Watsonx chat model you want to use (e.g., `ibm/granite-3-8b-instruct`).

**Example Code:**

Here's how to initialize and use Watsonx ChatModel:

```python
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import ToolMessage
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import MessageToolResultContent, UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Setting can be passed here during initiation or pre-configured via environment variables
llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    # settings={
    #     "project_id": "WATSONX_PROJECT_ID",
    #     "api_key": "WATSONX_API_KEY",
    #     "api_base": "WATSONX_API_URL",
    # },
)


async def watsonx_from_name() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "api_base": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

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
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def watson_tool_calling() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
    )
    user_message = UserMessage("What is the current weather in Boston?")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.create(messages=[user_message], tools=[weather_tool])
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await watsonx_llm.create(messages=[user_message, tool_response_msg], tools=[])
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
```

All providers examples can be found in [examples/backend/providers](/examples/backend/providers).

---

## Chat model

The ChatModel class represents a Chat Large Language Model and provides methods for text generation, streaming responses, and more. You can initialize a chat model in multiple ways:

**Method 1: Using the generic factory method**

```python
from beeai_framework.backend.chat import ChatModel

ollama_chat_model = ChatModel.from_name("ollama:llama3.1")
```

**Method 2: Creating a specific provider model directly**

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

ollama_chat_model = OllamaChatModel("llama3.1")
```

### Chat model configuration

You can configure various parameters for your chat model:

*Coming soon*

### Text generation

The most basic usage is to generate text responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

ollama_chat_model = OllamaChatModel("llama3.1")
response = await ollama_chat_model.create(
    messages=[UserMessage("what states are part of New England?")]
)

print(response.get_text_content())
```

> [!NOTE]
>
> Execution parameters (those passed to model.create({...})) are superior to ones defined via config.

### Streaming responses

For applications requiring real-time responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

llm = OllamaChatModel("llama3.1")
user_message = UserMessage("How many islands make up the country of Cape Verde?")
response = await llm.create(messages=[user_message], stream=True)
```

### Structured generation

Generate structured data according to a schema:

```python
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1")

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)
        address: str
        age: int = Field(..., min_length=1)
        hobby: str

    class ErrorSchema(BaseModel):
        error: str

    class SchemUnion(ProfileSchema, ErrorSchema):
        pass

    response = await model.create_structure(
        schema=SchemUnion,
        messages=[UserMessage("Generate a profile of a citizen of Europe.")],
    )

    print(
        json.dumps(
            response.object.model_dump() if isinstance(response.object, BaseModel) else response.object, indent=4
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

### Tool calling

Integrate external tools with your AI model:

```python
import asyncio
import json
import re
import sys
import traceback

from beeai_framework import Message, SystemMessage, Tool, ToolMessage, UserMessage
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.backend.message import MessageToolResultContent
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolOutput
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1", ChatModelParameters(temperature=0))
    tools: list[Tool] = [DuckDuckGoSearchTool(), OpenMeteoTool()]
    messages: list[Message] = [
        SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
        UserMessage("What's the fastest marathon time?"),
    ]

    while True:
        response = await model.create(
            messages=messages,
            tools=tools,
        )

        tool_calls = response.get_tool_calls()

        tool_results: list[ToolMessage] = []

        for tool_call in tool_calls:
            print(f"-> running '{tool_call.tool_name}' tool with {tool_call.args}")
            tool: Tool = next(tool for tool in tools if tool.name == tool_call.tool_name)
            assert tool is not None
            res: ToolOutput = await tool.run(json.loads(tool_call.args))
            result = res.get_text_content()
            print(f"<- got response from '{tool_call.tool_name}'", re.sub(r"\s+", " ", result)[:90] + " (truncated)")
            tool_results.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=result,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.id,
                    )
                )
            )

        messages.extend(tool_results)

        answer = response.get_text_content()

        if answer:
            print(f"Agent: {answer}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

---

## Embedding model

The EmbedingModel class provides functionality for generating vector embeddings from text.

### Embedding model initialization

You can initialize an embedding model in multiple ways:

**Method 1: Using the generic factory method**

*Coming soon*

**Method 2: Creating a specific provider model directly**

*Coming soon*

### Embedding model usage

Generate embeddings for one or more text strings:

*Coming soon*

---

## Advanced usage

If your preferred provider isn't directly supported, you can use the LangChain adapter as a bridge.

This allows you to leverage any provider that has LangChain compatibility.

*Coming soon*


_Source: /examples/backend/providers/langchain.py_

---

## Troubleshooting

Common issues and their solutions:

1. **Authentication errors:** Ensure all required environment variables are set correctly
2. **Model not found:** Verify that the model ID is correct and available for the selected provider

---

## Next Steps

Now that you have seen how to:
- Create prompt templates and render them dynamically.
- Interact with language models using ChatModel.
- Maintain conversation history with memory.
- Build structured output responses.
- Build a multi-step workflow (with and without memory).
- Configure a ReAct agent with custom and imported tools.
- **Orchestrate multi-agent systems using Workflows.**
- **Utilize Watsonx.ai backend within BeeAI Framework.**

You are well-equipped to start building sophisticated AI applications with the BeeAI framework. Explore the examples directory for more advanced use cases and integrations.  Dive deeper into the documentation to discover all the features and customization options BeeAI offers!

