# BeeAI Framework: Orchestrating Multi-Agent Systems for Security Event Analysis

Welcome to the BeeAI Framework! This guide focuses on using BeeAI to orchestrate multi-agent systems, specifically for analyzing security events. We will build a Proof of Concept (PoC) demo that showcases how BeeAI workflows can manage specialized agents to visualize and understand security data.

In this focused tutorial, we will cover:

- **Workflows for Orchestration:** How to use BeeAI workflows to manage multi-agent interactions.
- **Agent-Based System Design:** Creating specialized agents for research and visualization.
- **Tool Integration:** Using Python plotting libraries as tools within agents.
- **Backend Flexibility:** Demonstrating orchestration with both Ollama and Watsonx.

Let's dive into building a powerful security event analysis system using BeeAI!

---

## BeeAI Framework Core Concepts for Orchestration

For building our multi-agent system, understanding Workflows and AgentWorkflows is crucial.

### Workflows (Experimental)

Workflows in BeeAI provide a structured way to define and execute multi-step processes. They are ideal for orchestration because:

- **Dynamic Execution:** Steps can determine the next actions based on the current state or results.
- **Modularity:** Complex tasks can be broken down into reusable steps.
- **Observability:**  Execution can be tracked, and errors can be managed.

#### Core Concepts

- **State:** A Pydantic model holding data that is passed between workflow steps.
- **Steps:** Functions that perform actions, modify the state, and decide the next step.
- **Transitions:** Mechanisms for moving between steps based on step function returns (`Workflow.NEXT`, `Workflow.SELF`, `Workflow.END`, or step names).

### Agent Workflows

AgentWorkflows extend Workflows to specifically manage multiple agents within a workflow. This is perfect for orchestrating multi-agent systems where different agents with specialized roles collaborate.

---

## Building a Multi-Agent System for Security Event Analysis: PoC Demo

In this PoC, we will create a system to analyze security events from CSV data. The system will have three main components orchestrated by a BeeAI Workflow:

- **Data:** We will use three CSV files representing security events, asset relations, and asset knowledge base.
- **Agents:**
    - **Researcher Agent:**  Analyzes user queries and extracts relevant data subsets from the CSV data using Pandas.
    - **Visualizer Agent:**  Generates visualizations (graphs and plots) of the extracted data using tools.
    - **Orchestrator Agent (Workflow):** Manages the interaction between the Researcher, Visualizer, and the user, deciding which agent to invoke and when.
- **Tools:** Python functions for graph and interactive plotting using NetworkX, Matplotlib, and Plotly.

Let's start by setting up our environment and defining the necessary components.

### 1. Setting up the Environment and Data

First, ensure you have BeeAI Framework installed. For Watsonx support, install the Watsonx extension:

```bash
pip install beeai-framework[watsonx]
pip install pandas networkx matplotlib plotly scikit-learn
```

Next, let's load the sample CSV data into Pandas DataFrames.  Create three CSV files named `acme_corp_security_events.csv`, `acme_corp_relations.csv`, and `acme_corp_asset_kb.csv` in your project directory with the following content:

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

Now, load these CSV files in your Python script:

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

### 2. Defining Visualization Tools

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

### 3. Defining the Agents

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

### 4. Defining the Orchestrator Workflow

Now, we define the orchestrator workflow to manage the interaction between the agents and user.

```python
from beeai_framework.workflows.agent import AgentWorkflow
from beeai_framework.backend.message import UserMessage, AssistantMessage

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

### 5. Running the Orchestrated System with Ollama Backend

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
    asyncio.run(main_ollama_orchestration())
```

Run this script. You can ask questions like "show me high severity events" or "show me events from India".  After getting the dataframe preview, you will be asked to choose a plot type to visualize the data.

### 6. Orchestration with Watsonx.ai Backend

To use Watsonx as the backend, you only need to change the ChatModel initialization for both Researcher and Visualizer agents. Ensure you have set up your Watsonx environment variables as described in the Backend section of the full tutorial.

```python
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
    asyncio.run(main_watsonx_orchestration()) # Run Watsonx orchestration example

```

To run with Watsonx, execute the `main_watsonx_orchestration` function instead of `main_ollama_orchestration` in your `if __name__ == "__main__":` block.  Remember to configure your Watsonx API credentials and environment variables.

## Conclusion

This focused tutorial demonstrated how to build a multi-agent system for security event analysis using BeeAI Framework's workflow and agent capabilities. You learned how to define specialized agents, equip them with tools, and orchestrate their interactions using workflows.  By leveraging BeeAI, you can create sophisticated, modular, and flexible AI applications capable of complex data analysis and visualization, seamlessly switching between backends like Ollama and Watsonx.

For questions, discussions, or support, reach out to us via:

  * Email: [contact@ruslanmv.com](mailto:contact@ruslanmv.com)
  * GitHub Discussions: [BeeAI Framework Discussions](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/your-username/BeeAI-Framework-Practical-Guide/discussions)
  * Community Forum: [BeeAI Community](https://www.google.com/url?sa=E&source=gmail&q=https://community.beeai.org)

We sincerely thank our contributors, researchers, and supporters who have helped shape BeeAI. Special thanks to the open-source community for their invaluable feedback and contributions!
