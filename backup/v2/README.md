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
# (Same as previous code examples)
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
# Code as shown previously
```

---

## Building a Proof-of-Concept (PoC) Demo

We will build a PoC demo system that handles data from three CSV files: security events, relations, and asset knowledge bases. The orchestrator agent will:

1. Receive user queries.
2. Delegate analysis to a researcher agent to generate queries.
3. Extract relevant subsets using pandas.
4. Display the top 5 rows.
5. Provide interactive options to visualize:
   - Graph-based visualizations (using NetworkX)
   - Plotly-based interactive scatter plots

The visualization functions (`visualize_graph`, `visualize_node_distribution`, Plotly scatter plot) will provide insights based on user-selected criteria derived from the queried subsets.

This workflow allows users to quickly analyze complex cybersecurity data visually, leveraging BeeAI’s orchestration and multi-agent capabilities.

---

## Conclusion

You've now learned how to orchestrate sophisticated workflows and multi-agent systems within the BeeAI Framework using both Ollama and Watsonx.ai. You can apply these patterns to build advanced conversational agents and integrate various AI services seamlessly.

Happy coding! 🐝

