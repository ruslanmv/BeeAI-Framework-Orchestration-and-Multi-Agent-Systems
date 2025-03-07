# 🐝 BeeAI Framework: Orchestration and Multi-Agent Systems

This tutorial builds upon our previous guide, diving deeper into workflows and orchestration within the BeeAI Framework. By the end of this tutorial, you'll be proficient in creating multi-agent systems and orchestrating workflows using both Ollama and Watsonx.ai.

## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [State](#state)
  - [Steps](#steps)
  - [Transitions](#transitions)
- [Basic Usage](#basic-usage)
  - [Simple Workflow](#simple-workflow)
  - [Multi-Step Workflow](#multi-step-workflow)
- [Advanced Orchestration](#advanced-orchestration)
  - [Multi-Agent Workflows](#multi-agent-workflows)
  - [Orchestrating with Ollama](#orchestrating-with-ollama)
  - [Orchestrating with Watsonx.ai](#orchestrating-with-watsonxai)

---

## Overview

BeeAI workflows support dynamic task execution, validation, modularity, and observability, providing the foundation for creating sophisticated orchestrations.

---

## Core Concepts

### State
A structured representation of data flowing between workflow steps, validated using Pydantic schemas.

### Steps
Functions executed sequentially or conditionally within a workflow, manipulating state data and guiding workflow transitions.

### Transitions
Determine the logical flow between workflow steps, directing which step executes next or signaling workflow completion.

---

## Basic Usage

### Simple Workflow
A straightforward sequence of operations:
```python
from pydantic import BaseModel
from beeai_framework.workflows.workflow import Workflow

class State(BaseModel):
    message: str

async def first_step(state: State):
    state.message += " BeeAI"
    print("First Step executed")
    return Workflow.NEXT

async def second_step(state: State):
    state.message += " Framework"
    print("Second Step executed")
    return Workflow.END

workflow = Workflow(schema=State)
workflow.add_step("first_step", first_step)
workflow.add_step("second_step", second_step)

response = await workflow.run(State(message="Hello"))
print(response.state.message)
```

### Multi-Step Workflow
Demonstrates conditional logic and loops within workflows:
```python
from pydantic import BaseModel
from beeai_framework.workflows.workflow import Workflow

class CalcState(BaseModel):
    x: int
    y: int
    result: int = 0

async def multiply(state: CalcState):
    for _ in range(abs(state.y)):
        state.result += state.x
    state.result = state.result if state.y >= 0 else -state.result
    return Workflow.END

calc_workflow = Workflow(schema=CalcState)
calc_workflow.add_step("multiply", multiply)

response = await calc_workflow.run(CalcState(x=7, y=-3))
print(response.state.result)
```

---

## Advanced Orchestration

### Multi-Agent Workflows
Integrate specialized agents for complex tasks:

```python
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow

async def main():
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Assistant")
    workflow.add_agent("WeatherAgent", instructions="Fetch weather data.", tools=[OpenMeteoTool()], llm=llm)
    workflow.add_agent("SearchAgent", instructions="Perform web searches.", tools=[DuckDuckGoSearchTool()], llm=llm)
    workflow.add_agent("Coordinator", instructions="Summarize responses.", llm=llm)

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Weather and events in Paris?"))

    response = await workflow.run(messages=memory.messages)
    print(f"Final Answer: {response.state.final_answer}")
```

### Orchestrating with Ollama
Using Ollama provider for orchestration:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

async def ollama_example():
    llm = OllamaChatModel("llama3.1")
    user_message = UserMessage("Describe the Eiffel Tower.")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())

await ollama_example()
```

### Orchestrating with Watsonx.ai
Integration with Watsonx.ai for orchestration:

```python
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.message import UserMessage

async def watsonx_example():
    watsonx_llm = WatsonxChatModel("ibm/granite-3-8b-instruct")
    user_message = UserMessage("Historical significance of the Eiffel Tower?")
    response = await watsonx_llm.create(messages=[user_message])
    print(response.get_text_content())

await watsonx_example()
```

### Complex Multi-Agent Orchestration (Watsonx and Ollama combined)
Combine multiple models into a single orchestration:

```python
from beeai_framework.workflows.agent import AgentWorkflow
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

async def hybrid_orchestration():
    ollama_llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    watsonx_llm = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct")

    workflow = AgentWorkflow(name="HybridAssistant")
    workflow.add_agent("HistoricalAgent", instructions="Answer historical questions.", tools=[DuckDuckGoSearchTool()], llm=watsonx_llm)
    workflow.add_agent("GeneralAgent", instructions="Answer general questions.", llm=ollama_llm)
    workflow.add_agent("Coordinator", instructions="Integrate agent responses.", llm=ollama_llm)

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Give historical and general information about the Eiffel Tower."))

    response = await workflow.run(messages=memory.messages)
    print(f"Integrated Answer: {response.state.final_answer}")

await hybrid_orchestration()
```

---

## Conclusion

You've now learned how to orchestrate sophisticated workflows and multi-agent systems within the BeeAI Framework using both Ollama and Watsonx.ai. You can apply these patterns to build advanced conversational agents and integrate various AI services seamlessly.


