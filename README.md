# 🐝 BeeAI Framework: Orchestration and Multi-Agent Systems

This tutorial builds upon our previous guides, diving deeper into workflows and orchestration within the BeeAI Framework.  We will explore how to leverage BeeAI's powerful workflow engine to construct sophisticated multi-agent systems. By the end of this tutorial, you'll be proficient in creating multi-agent systems and orchestrating workflows using both Ollama and Watsonx.ai backends. This guide culminates in building a practical Proof of Concept (PoC) demo for security event analysis, showcasing the real-world applicability of BeeAI.

## Table of Contents

-   [Overview](#overview)
-   [Setup](#setup)
-   [Core Concepts](#core-concepts)
    -   [State](#state)
    -   [Steps](#steps)
    -   [Transitions](#transitions)
-   [Basic Usage](#basic-usage)
    -   [Simple Workflow](#simple-workflow)
    -   [Multi-Step Workflow](#multi-step-workflow)
-   [Advanced Orchestration](#advanced-orchestration)
    -   [Multi-Agent Workflows](#multi-agent-workflows)
    -   [Orchestrating with Ollama](#orchestrating-with-ollama)
    -   [Orchestrating with Watsonx.ai](#orchestrating-with-watsonxai)
-   [Building a Proof-of-Concept (PoC) Demo](#building-a-proof-of-concept-poc-demo)
    -   [1. Setting up the Environment and Data](#1-setting-up-the-environment-and-data)
    -   [2. Defining Visualization Tools](#2-defining-visualization-tools)
    -   [3. Defining the Agents](#3-defining-the-agents)
    -   [4. Defining the Orchestrator Workflow](#4-defining-the-orchestrator-workflow)
    -   [5. Running the Orchestrated System with Ollama Backend](#5-running-the-orchestrated-system-with-ollama-backend)
    -   [6. Orchestration with Watsonx.ai Backend](#6-orchestration-with-watsonxai-backend)
-   [Conclusion](#conclusion)

## Overview

BeeAI workflows support dynamic task execution, validation, modularity, and observability, providing the foundation for creating sophisticated orchestrations.  They enable developers to design complex, multi-step processes where each step is clearly defined, data flow is managed, and execution paths can dynamically adapt based on outcomes or conditions.  This makes BeeAI particularly powerful for building intelligent agents and systems that need to perform complex reasoning and actions.

## Setup

Follow these steps to set up your environment for running BeeAI Framework on Windows or Ubuntu 22.04.

## Prerequisites

- **Python 3.12+** (Ensure it is added to PATH during installation)
- **Anaconda or Miniconda** (Recommended for environment management)

## Windows Setup

1. **Install Python 3.12+**: Download from [python.org](https://www.python.org/downloads/windows/).

2. **Install Anaconda/Miniconda**: Download from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. Open Anaconda Prompt

    and create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install BeeAI Framework and dependencies:

   ```bash
   pip install beeai-framework pandas networkx matplotlib plotly scikit-learn
   ```

5. Install and Start Ollama:

   - Download from [ollama.com](https://ollama.com/download) and install.

   - Start Ollama server:

     ```bash
     ollama serve &
     ```

   - Download the required model:

     ```bash
     ollama pull granite3.1-dense:8b
     ```

6. Set Watsonx.ai Credentials (If required):

   ```bash
   set WATSONX_PROJECT_ID=YOUR_WATSONX_PROJECT_ID
   set WATSONX_API_KEY=YOUR_WATSONX_API_KEY
   set WATSONX_API_URL=YOUR_WATSONX_API_ENDPOINT_URL
   ```

## Ubuntu 22.04 Setup

1. Install Python 3.12+:

   ```bash
   sudo apt update && sudo apt install python3.12 python3.12-venv
   ```

2. Install Anaconda/Miniconda:

   - Download from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run the installer.
   - Reload shell: `source ~/.bashrc` or `source ~/.zshrc`

3. Create and activate virtual environment:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

4. Install BeeAI Framework and dependencies:

   ```bash
   pip install beeai-framework pandas networkx matplotlib plotly scikit-learn
   ```

5. Install and Start Ollama:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull granite3.1-dense:8b
   ```

6. Set Watsonx.ai Credentials (If required):

   ```bash
   export WATSONX_PROJECT_ID=YOUR_WATSONX_PROJECT_ID
   export WATSONX_API_KEY=YOUR_WATSONX_API_KEY
   export WATSONX_API_URL=YOUR_WATSONX_API_ENDPOINT_URL
   ```

## Notes

- Always activate your virtual environment before running BeeAI.
- Ensure Ollama server is running in the background.
- Use environment variables to manage Watsonx credentials securely.
- For troubleshooting, refer to BeeAI documentation.

## Core Concepts

### State

State is the central element of a BeeAI workflow. It is a structured representation of data flowing between workflow steps.  Defined using Pydantic schemas, the state ensures data consistency and type safety throughout the workflow execution. This robust data management is essential for reliable and predictable behavior in complex orchestrations.  The state acts as a shared memory space, allowing each step to access and modify data in a controlled and type-validated manner.

### Steps

Steps are the building blocks of a workflow. They are functions that are executed sequentially or conditionally within a workflow. Each step takes the current state as input, performs a specific task—such as data processing, agent invocation, or tool execution—and then manipulates the state data. Crucially, steps also guide workflow transitions by returning specific signals that dictate which step executes next or when the workflow completes.  Steps provide modularity and allow complex processes to be broken down into manageable, testable units.

### Transitions

Transitions determine the logical flow between workflow steps. Each step function returns a transition directive that directs the workflow engine. These transitions can be:

  - `Workflow.NEXT`:  Proceed to the next step in the sequence as defined in the workflow.
  - `Workflow.SELF`: Repeat the current step, often used to create loops or iterative processes.
  - `Workflow.END`: Signal the completion of the workflow.
  - Step Name (String):  Jump to a specific step identified by its name, enabling conditional branching and non-linear workflows.

Transitions are critical for creating dynamic and responsive orchestrations, allowing workflows to adapt to different conditions and outcomes during runtime.

## Basic Usage

### Simple Workflow

A straightforward sequence of operations demonstrating basic workflow structure:

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

In this example, `first_step` and `second_step` are executed in order. The `State` Pydantic model ensures that the `message` attribute is correctly passed and modified between steps, illustrating a linear workflow progression.

### Multi-Step Workflow

Demonstrating conditional logic and loops within workflows, this example implements multiplication using repeated addition:

```python
from pydantic import BaseModel
from beeai_framework.workflows.workflow import Workflow

class CalcState(BaseModel):
    x: int
    y: int
    result: int = 0

async def multiply(state: CalcState):
    if state.y > 0:
        for _ in range(state.y):
            state.result += state.x
    elif state.y < 0:
        for _ in range(abs(state.y)):
            state.result -= state.x
    return Workflow.END

calc_workflow = Workflow(schema=CalcState)
calc_workflow.add_step("multiply", multiply)

response = await calc_workflow.run(CalcState(x=7, y=-3))
print(response.state.result)
```

This workflow uses conditional logic within the `multiply` step to handle both positive and negative multipliers, showcasing how workflows can implement more complex logic beyond simple sequential execution.

## Advanced Orchestration

### Multi-Agent Workflows

BeeAI excels at orchestrating multi-agent systems, allowing you to integrate specialized agents to tackle complex tasks collaboratively.  An `AgentWorkflow` is specifically designed to manage and coordinate multiple `BeeAgent` instances, enabling the creation of sophisticated, modular agent-based applications. This section will guide you through building such a system.

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
    llm_watsonx = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct") # Replace with your Watsonx model
    llm_ollama = ChatModel.from_name("ollama:llama3.1")


    workflow = AgentWorkflow(name="Smart assistant (Mixed Backend)")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant.",
        tools=[OpenMeteoTool()],
        llm=llm_watsonx, # Using Watsonx for WeatherForecaster
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm_ollama, # Using Ollama for Researcher
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm_watsonx, # Using Watsonx for Solver
    )

    prompt = "What is the weather in London and the capital of France?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result (Mixed Backend): {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

## Building a Proof-of-Concept (PoC) Demo

We will build a PoC demo system that handles data from three CSV files: security events, relations, and asset knowledge bases. The orchestrator agent will manage a workflow to:

1.  **Receive user queries.**
2.  **Delegate analysis to a researcher agent** to generate pandas queries and extract relevant data.
3.  **Extract relevant subsets using pandas.**
4.  **Display the top 5 rows of the extracted dataframe.**
5.  **Provide interactive options to visualize the data**, offering:
      - Graph-based visualizations (using NetworkX) for relationship analysis.
      - Plotly-based interactive scatter plots for time-series event density and severity.

The visualization functions (`visualize_graph`, `visualize_node_distribution`, Plotly scatter plot) will provide insights based on user-selected criteria derived from the queried subsets. This workflow allows users to quickly analyze complex cybersecurity data visually, leveraging BeeAI’s orchestration and multi-agent capabilities.

### 1\. Setting up the Environment and Data

First, ensure you have BeeAI Framework installed. For Watsonx support and visualization tools, install the necessary extensions and libraries:

```bash
pip install beeai-framework[watsonx]
pip install pandas networkx matplotlib plotly scikit-learn
```

Next, create three CSV files named `acme_corp_security_events.csv`, `acme_corp_relations.csv`, and `acme_corp_asset_kb.csv` in your project directory with the following content. These files represent sample security data for Acme Corp.

**`acme_corp_security_events.csv`:**

```csv
event_timestamp,event_type,event_severity,threat_actor,attack_vector,event_category,DateOrdinal,event_density,asset_ip_range,asset_isp,asset_country,asset_location,asset_type,asset_criticality
2023-01-21,Phishing Email Detected,High,Cybercriminal,Exploit,Access Violation,738541,0.001623,175.205.130.0/24,SecureCloud,India,Stockholm,Mobile Device,Medium
2023-01-04,Vulnerability Exploitation Attempt,Low,Insider,Brute Force,Performance Issue,738524,0.001314,94.247.229.0/24,DataStream,Canada,Los Angeles,Mobile Device,Medium
2023-01-01,Suspicious Login Activity,Medium,Unknown,Exploit,Security Incident,738521,0.001278,203.117.143.0/16,ApexNet,Germany,London,IoT Device,Medium
```

**`acme_corp_relations.csv`:**

```csv
source,target,relation
103.179.92.0/24,ApexNet,belongs_to_isp
236.88.48.0/24,InfiNet,belongs_to_isp
130.83.91.0/16,NetSphere,belongs_to_isp
```

**`acme_corp_asset_kb.csv`:**

```csv
ip_range,isp,country,location,asset_type,criticality,last_activity,vulnerabilities,data_classification,network_zone,connected_assets,notes,operating_system,installed_software,department,compliance_status
103.179.92.0/24,ApexNet,Switzerland,Sydney,Mobile Device,Critical,2025-02-25,7,Internal,Management Network,23,NaN,Linux,"['Development Tools']",Marketing,Non-Compliant
236.88.48.0/24,InfiNet,Netherlands,Chicago,Server,Critical,2025-01-16,6,Restricted,DMZ,2,NaN,iOS,"['Cloud Service Client', 'Database', 'System U...]",Engineering,Compliant
130.83.91.0/16,NetSphere,Netherlands,Singapore,Workstation,Low,2025-01-31,1,Confidential,Internal Network,5,NaN,Windows,"['Office Suite', 'Database']",Operations,Non-Compliant
```

Load these CSV files into Pandas DataFrames in your Python script:

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.stats import gaussian_kde
import random

# Load CSV data
df_security_events = pd.read_csv("./acme_corp_security_events.csv")
df_asset_kb = pd.read_csv("./acme_corp_asset_kb.csv")
df_relations = pd.read_csv("./acme_corp_relations.csv")

# Prepare relations data for graph visualization
database_with_relations = df_relations.to_dict('records')

# Prepare start date for Plotly plot
start_date = datetime(2023, 1, 1)
```

### 2\. Defining Visualization Tools

We will define Python functions as tools for our agents. These tools will handle graph visualizations and interactive plots.

```python
# ---------------- Graph Visualization Functions ----------------

# Function to construct graph from database
def build_graph(database):
    G = nx.DiGraph()
    for entry in database:
        G.add_edge(entry["source"], entry["target"], relation=entry["relation"])
    return G

# Visualize function
def visualize_graph(graph, title="Knowledge Graph Visualization"):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(graph, seed=42)
    labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=4000, font_size=10, alpha=0.7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=9)
    plt.title(title, fontsize=14)
    plt.show()

# Function to visualize node distribution
def visualize_node_distribution(database, relation_type, title_suffix):
    filtered_relations = [entry for entry in database if entry["relation"] == relation_type]
    subgraph = build_graph(filtered_relations)
    visualize_graph(subgraph, title=f"Distribution of Entities - {title_suffix}")

    node_counts = {}
    for entry in filtered_relations:
        target_node = entry["target"]
        node_counts[target_node] = node_counts.get(target_node, 0) + 1

    plt.figure(figsize=(10, 6))
    plt.bar(node_counts.keys(), node_counts.values(), color='skyblue')
    plt.xlabel(title_suffix.split("by")[-1].strip(), fontsize=12)
    plt.ylabel("Number of Assets/Events", fontsize=12)
    plt.title(f"Distribution of Assets/Events {title_suffix}", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ---------------- Plotly Interactive Plot Function ----------------

def create_plotly_timeline(df_security_events_in, start_date_in):
    df_plotly = df_security_events_in.copy()
    df_plotly = df_plotly.rename(columns={"event_timestamp": "Date", "event_type": "Label", "event_density": "DensityY", "event_severity": "Intensity"})
    df_plotly["Intensity"] = df_plotly["Intensity"].astype('category').cat.codes + 1
    df_plotly["DateOrdinal"] = df_plotly["Date"].map(datetime.toordinal)

    date_ordinals = df_plotly["DateOrdinal"].values
    kde = gaussian_kde(date_ordinals, bw_method=0.5)
    x_range = np.linspace(start_date_in.toordinal(), datetime(2024, 5, 1).toordinal(), 500)
    kde_values = kde(x_range)
    max_density = kde_values.max()

    scatter_y = []
    for date_val in df_plotly["DateOrdinal"]:
        idx = np.searchsorted(x_range, date_val)
        local_density = kde_values[max(0, min(idx, len(kde_values)-1))] # Safe index access
        offset = random.uniform(-0.1, 0.1) * (max_density / 3.0)
        y_val  = max(0, local_density + offset)
        scatter_y.append(y_val)

    df_plotly["DensityY"] = scatter_y

    fig_plotly = px.scatter(
        df_plotly,
        x="Date",
        y="DensityY",
        color="Label",
        size="Intensity",
        size_max=20,
        opacity=0.7,
        labels={"DensityY": "Estimated Event Density", "Date": "Time"},
        title="Interactive Timeline of Security Events with Density and Severity"
    )

    x_dt = [datetime.fromordinal(int(xo)) for xo in x_range]
    kde_trace = go.Scatter(
        x=x_dt,
        y=kde_values,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(128,128,128,0.3)',
        line=dict(color='gray'),
        name='KDE Density'
    )
    fig_plotly.add_trace(kde_trace)

    fig_plotly.update_layout(
        xaxis_title="Time",
        yaxis_title="Estimated Event Density",
        title="Interactive Timeline of Security Events with KDE Fill + Density and Severity",
        legend_title="Event Type / Severity",
        xaxis=dict(
            tickmode='array',
            tickvals=[datetime(2023 + (y>0), (y or 1) if (y or 1) <= 12 else (y % 12 or 12) , 1) for y in range(0, 14)],
            ticktext=[datetime(2023 + (y>0), (y or 1) if (y or 1) <= 12 else (y % 12 or 12) , 1).strftime('%Y-%m') for y in range(0, 14)],
            range=[start_date_in, datetime(2024, 5, 1)]
        ),
        yaxis=dict(
            zeroline=False,
            title_standoff=10
        ),
        plot_bgcolor='white',
        margin=dict(r=200)
    )
    fig_plotly.show()
```

### 3\. Defining the Agents

Now, let's define our agents: `ResearcherAgent` and `VisualizerAgent`.  We will use Ollama backend for this example, ensure you have Ollama running and accessible.

```python
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools import Tool, tool
from pydantic import BaseModel, Field
from typing import List

# Initialize Ollama ChatModel
ollama_llm = ChatModel.from_name("ollama:llama3.1")

# -------------- Researcher Agent and Tool ------------------

class DataQuerySchema(BaseModel):
    pandas_query: str = Field(description="Pandas query to filter the dataframe.")

class ResearcherToolInput(BaseModel):
    query: str = Field(description="User query to analyze security events data.")

class ResearcherToolOutput(BaseModel):
    dataframe_head: str = Field(description="First 5 rows of the filtered dataframe in markdown format.")

class ResearcherAgent:
    @tool(input_schema=ResearcherToolInput, output_schema=ResearcherToolOutput)
    async def analyze_data(query: str) -> ResearcherToolOutput:
        """
        Analyzes security events data based on user query and returns a dataframe preview.
        """
        try:
            # Placeholder for generating pandas query from natural language query (simple example)
            pandas_query = f"df_security_events[{query}]" # In real use case, use LLM to generate safe queries.
            filtered_df = eval(pandas_query) # WARNING: Be extremely cautious with eval in production. Use safer methods.
            dataframe_head_md = filtered_df.head().to_markdown()
            return ResearcherToolOutput(dataframe_head=dataframe_head_md)
        except Exception as e:
            return ResearcherToolOutput(dataframe_head=f"Error executing query: {e}")

# -------------- Visualizer Agent and Tools ------------------

class PlotType(BaseModel):
    plot_name: str = Field(enum=["graph_visualization", "node_distribution", "plotly_timeline"], description="Type of plot to generate.")

class VisualizationToolInput(BaseModel):
    plot_type: PlotType = Field(description="Type of plot to generate.")

class VisualizationToolOutput(BaseModel):
    plot_message: str = Field(description="Message indicating plot generation.")

class VisualizerAgent:
    @tool(input_schema=VisualizationToolInput, output_schema=VisualizationToolOutput)
    async def generate_visualization(plot_type: PlotType) -> VisualizationToolOutput:
        """
        Generates visualizations based on the analyzed security events data.
        """
        plot_name = plot_type.plot_name
        if plot_name == "graph_visualization":
            visualize_graph(build_graph(database_with_relations), title="Knowledge Graph")
        elif plot_name == "node_distribution":
            visualize_node_distribution(database_with_relations, "located_in_country", "Assets by Country")
        elif plot_name == "plotly_timeline":
            create_plotly_timeline(df_security_events, start_date)
        return VisualizationToolOutput(plot_message=f"Generated {plot_name} plot.")


# -------------- Instantiate Agents ------------------
researcher_agent_instance = ResearcherAgent()
visualizer_agent_instance = VisualizerAgent()

researcher_bee_agent = BeeAgent(bee_input=BeeInput(
    llm=ollama_llm,
    tools=[researcher_agent_instance.analyze_data],
    memory=UnconstrainedMemory(),
    instructions="You are a data researcher. Use pandas to query and analyze security event dataframes."
))

visualizer_bee_agent = BeeAgent(bee_input=BeeInput(
    llm=ollama_llm,
    tools=[visualizer_agent_instance.generate_visualization],
    memory=UnconstrainedMemory(),
    instructions="You are a data visualizer. Generate plots and graphs to represent data."
))
```

**Important Security Note:** Be extremely cautious when using `eval()` with user-provided input in a production environment.  This PoC uses it for simplicity in demonstrating the concept, but in a real-world application, you must implement secure query generation and execution to prevent injection vulnerabilities. Consider using parameterized queries or an ORM to safely interact with data.

### 4\. Defining the Orchestrator Workflow

Now, we define the orchestrator workflow to manage the interaction between the agents and user.

```python
from beeai_framework.workflows.agent import AgentWorkflow
from beeai_framework.backend.message import UserMessage, AssistantMessage
from pydantic import BaseModel, Field
from typing import List
from beeai_framework.agents.types import BeeRunOutput, BeeRunInput
from beeai_framework.workflows.workflow import Workflow

class AnalysisState(BaseModel):
    query: str = Field(description="User's query for security event analysis.")
    researcher_output: str = Field(default=None, description="Output from the Researcher Agent.")
    plot_options: List[str] = Field(default=["graph_visualization", "node_distribution", "plotly_timeline"], description="Available plot types.")
    visualizer_output: str = Field(default=None, description="Output from Visualizer Agent after plotting.")
    final_answer: str = Field(default=None, description="Final summarized answer to the user.")

async def ask_researcher_step(state: AnalysisState) -> str:
    print("Orchestrator Step: Ask Researcher")
    researcher_result: BeeRunOutput = await orchestrator_workflow.agents['Researcher'].run(
        run_input=BeeRunInput(prompt=state.query)
    )
    state.researcher_output = researcher_result.state.agent_response.get_text_content()
    return "display_dataframe_step"

async def display_dataframe_step(state: AnalysisState) -> str:
    print("Orchestrator Step: Display Dataframe Preview")
    print("Researcher Output (Dataframe Preview):\n", state.researcher_output)
    return "ask_visualization_preference_step"

async def ask_visualization_preference_step(state: AnalysisState) -> str:
    print("Orchestrator Step: Ask Visualization Preference")
    plot_options_str = ", ".join(state.plot_options)
    state.final_answer = f"Data preview displayed. Available plot options: {plot_options_str}. Which plot would you like to generate? (or type 'none' for no plot)"
    return Workflow.END # For interactive demo, we will ask user manually, in real app, next step can be another agent.


async def generate_visualization_step(state: AnalysisState, plot_name: str) -> str: # Passing plot_name as parameter
    print(f"Orchestrator Step: Generate Visualization - {plot_name}")
    visualizer_result: BeeRunOutput = await orchestrator_workflow.agents['Visualizer'].run(
        run_input=BeeRunInput(prompt=plot_name, input_parameters={"plot_type": {"plot_name": plot_name}}) # Pass plot_type dynamically
    )
    state.visualizer_output = visualizer_result.state.agent_response.get_text_content()
    print("Visualizer Output:\n", state.visualizer_output)
    state.final_answer = "Visualization generated and displayed." # Update final answer
    return Workflow.END

# Initialize AgentWorkflow for Orchestration
orchestrator_workflow = AgentWorkflow(schema=AnalysisState, name="SecurityDataAnalysisWorkflow")

# Add Agents to the Workflow
orchestrator_workflow.add_agent(name="Researcher", agent=researcher_bee_agent, is_coordinator=False)
orchestrator_workflow.add_agent(name="Visualizer", agent=visualizer_bee_agent, is_coordinator=False)

# Add Orchestration Steps
orchestrator_workflow.add_step("ask_researcher_step", ask_researcher_step)
orchestrator_workflow.add_step("display_dataframe_step", display_dataframe_step)
orchestrator_workflow.add_step("ask_visualization_preference_step", ask_visualization_preference_step)
```

### 5\. Running the Orchestrated System with Ollama Backend

Now, we can run the orchestrated system. This example uses Ollama as the backend.

```python
async def main_ollama_orchestration():
    # Start interactive session
    while True:
        user_query = input("User Query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # Initialize state for each run
        analysis_state = AnalysisState(query=user_query, researcher_output=None, visualizer_output=None, final_answer=None)

        # Run the workflow up to dataframe display
        research_response = await orchestrator_workflow.run(analysis_state, steps=["ask_researcher_step", "display_dataframe_step", "ask_visualization_preference_step"])
        print("Assistant:", research_response.state.final_answer)


        while True:
            plot_choice = input(f"Choose plot type from {research_response.state.plot_options} or 'none': ")
            if plot_choice.lower() == 'none':
                break
            if plot_choice in research_response.state.plot_options:
                visualization_state = AnalysisState(query=user_query) # Re-initiate state, or carry over as needed
                visualization_state.researcher_output = research_response.state.researcher_output # Carry over dataframe if needed
                plot_response = await generate_visualization_step(visualization_state, plot_choice) # Call plot step directly
                print("Assistant:", visualization_state.final_answer)
                break
            else:
                print("Invalid plot choice. Please choose from available options or 'none'.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_ollama_orchestration())
```

Run this script. You can ask questions like "show me high severity events" or "show me events from India".  After getting the dataframe preview, you will be asked to choose a plot type to visualize the data.

### 6\. Orchestration with Watsonx.ai Backend

To use Watsonx as the backend, you only need to change the ChatModel initialization for both Researcher and Visualizer agents. Ensure you have set up your Watsonx environment variables as described in the Backend section of the full tutorial.

```python
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools import Tool, tool
from pydantic import BaseModel, Field
from typing import List

# Initialize Watsonx ChatModel
watsonx_llm = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct") # Replace with your Watsonx model as needed

researcher_bee_agent_watsonx = BeeAgent(bee_input=BeeInput(
    llm=watsonx_llm, # Use Watsonx LLM
    tools=[researcher_agent_instance.analyze_data],
    memory=UnconstrainedMemory(),
    instructions="You are a data researcher. Use pandas to query and analyze security event dataframes."
))

visualizer_bee_agent_watsonx = BeeAgent(bee_input=BeeInput(
    llm=watsonx_llm, # Use Watsonx LLM
    tools=[visualizer_agent_instance.generate_visualization],
    memory=UnconstrainedMemory(),
    instructions="You are a data visualizer. Generate plots and graphs to represent data."
))

# Re-define the orchestrator workflow with Watsonx backed agents
orchestrator_workflow_watsonx = AgentWorkflow(schema=AnalysisState, name="SecurityDataAnalysisWorkflowWatsonx")
orchestrator_workflow_watsonx.add_agent(name="Researcher", agent=researcher_bee_agent_watsonx, is_coordinator=False)
orchestrator_workflow_watsonx.add_agent(name="Visualizer", agent=visualizer_bee_agent_watsonx, is_coordinator=False)
orchestrator_workflow_watsonx.add_step("ask_researcher_step", ask_researcher_step)
orchestrator_workflow_watsonx.add_step("display_dataframe_step", display_dataframe_step)
orchestrator_workflow_watsonx.add_step("ask_visualization_preference_step", ask_visualization_preference_step)


async def main_watsonx_orchestration():
    # Start interactive session with Watsonx backend
    while True:
        user_query = input("User Query (Watsonx Backend, or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # Initialize state for each run
        analysis_state = AnalysisState(query=user_query, researcher_output=None, visualizer_output=None, final_answer=None)

        # Run the workflow with Watsonx agents
        research_response = await orchestrator_workflow_watsonx.run(analysis_state, steps=["ask_researcher_step", "display_dataframe_step", "ask_visualization_preference_step"])
        print("Assistant (Watsonx):", research_response.state.final_answer)

        while True:
            plot_choice = input(f"Choose plot type from {research_response.state.plot_options} or 'none': ")
            if plot_choice.lower() == 'none':
                break
            if plot_choice in research_response.state.plot_options:
                visualization_state = AnalysisState(query=user_query) # Re-initiate state, or carry over as needed
                visualization_state.researcher_output = research_response.state.researcher_output # Carry over dataframe if needed
                plot_response = await generate_visualization_step(visualization_state, plot_choice) # Call plot step directly
                print("Assistant:", visualization_state.final_answer)
                break
            else:
                print("Invalid plot choice. Please choose from available options or 'none'.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_watsonx_orchestration()) # To run with Watsonx, uncomment this and comment out Ollama main
    # asyncio.run(main_ollama_orchestration()) # To run with Ollama, uncomment this and comment out Watsonx main
```

To run with Watsonx, execute the `main_watsonx_orchestration` function instead of `main_ollama_orchestration` in your `if __name__ == "__main__":` block.  Remember to configure your Watsonx API credentials and environment variables.

## Conclusion

This tutorial has provided a comprehensive guide to building a multi-agent system using BeeAI Framework, focused on a practical security event analysis PoC.  You have seen how to:

  - Structure workflows to orchestrate complex tasks.
  - Create specialized agents for research and visualization.
  - Integrate Python plotting tools within agents.
  - Run the same orchestration seamlessly with different backends like Ollama and Watsonx.ai.

BeeAI’s workflow and agent orchestration capabilities offer a powerful and flexible platform for building sophisticated AI applications. By breaking down complex tasks into manageable steps and specialized agents, and leveraging the dynamic execution and modularity of BeeAI workflows, you can create robust and adaptable AI solutions. This PoC demonstrates just a glimpse of the potential for BeeAI in data analysis, visualization, and beyond.

For questions, discussions, or support, reach out to us via:

  \* Email: [contact@ruslanmv.com](mailto:contact@ruslanmv.com)
  \* GitHub Discussions: [BeeAI Framework Discussions](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://github.com/your-username/BeeAI-Framework-Practical-Guide/discussions)
  \* Community Forum: [BeeAI Community](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://community.beeai.org)

We sincerely thank our contributors, researchers, and supporters who have helped shape BeeAI. Special thanks to the open-source community for their invaluable feedback and contributions\!