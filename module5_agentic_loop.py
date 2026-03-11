import os
import json
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# ---------------------------------------------------------
# 1. Define the Tools (The Agent's Hands)
# ---------------------------------------------------------
# We use the @tool decorator from LangChain. It automatically extracts
# the name, description, and parameter types from the Python function!
@tool
def check_weather(location: str) -> str:
    """Mock weather tool."""
    mock_db = {"london": "Rainy", "tokyo": "Sunny", "paris": "Cloudy"}
    return mock_db.get(location.lower(), f"Weather data not found for {location}.")

@tool
def calculate_math(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

# List of tools our agent is allowed to use
tools = [check_weather, calculate_math]

# ---------------------------------------------------------
# 2. Define State & Initialize the LLM
# ---------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM and bind the tools to it
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# 3. Define the Nodes
# ---------------------------------------------------------

def agent_node(state: State):
    """The Agent's brain. Decides whether to answer or use a tool."""
    messages = state["messages"]
    if len(messages) == 1:
        # Inject system prompt on first turn
        sys_msg = SystemMessage(content="You are a helpful assistant with access to tools. Use them when needed.")
        messages = [sys_msg] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# LangGraph provides a pre-built ToolNode that automatically executes 
# the Python functions if the LLM requested them.
tool_executor_node = ToolNode(tools)

# ---------------------------------------------------------
# 4. Build the Graph (The Agentic Loop)
# ---------------------------------------------------------
graph_builder = StateGraph(State)

# Add our two nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_executor_node)

# Step 1: Start at the agent
graph_builder.add_edge(START, "agent")

# Step 2: After the agent thinks, we use a CONDITIONAL edge
# The built-in `tools_condition` checks the agent's last message:
# - If it asked for a tool -> go to "tools"
# - If it just replied normally -> go to END
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# Step 3: After the tool runs, ALWAYS go back to the agent so it can read the result!
graph_builder.add_edge("tools", "agent")

# Compile the graph
# In this module, we add persistent memory by using a MemorySaver checkpointer.
# This saves the graph state locally between turns, so we don't have to manually 
# keep track of `session_messages` like we did in Module 4.
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# ---------------------------------------------------------
# 5. Run the Agent
# ---------------------------------------------------------
def main():
    print("🧠 Welcome to Module 5: The Agentic Loop (Memory & Planning)!")
    print("This agent runs in a continuous loop until it has an answer.")
    print("Type 'exit' to quit.\n")

    # We define a generic 'thread_id' config to uniquely identify this conversation
    # inside our MemorySaver checkpointer.
    config = {"configurable": {"thread_id": "session_1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        print("\n--- Agent internal thought process ---")
        
        # We pass the user input as a standard tuple. Because we have a Checkpointer,
        # LangGraph automatically loads the exact State from the last loop before appending this!
        events = app.stream(
            {"messages": [("user", user_input)]}, 
            config, 
            stream_mode="values"
        )
        
        for event in events:
            # We look at the very last message added to the state in the current step
            last_msg = event["messages"][-1]
            print(last_msg)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🔧 Agent decided to call tool: {last_msg.tool_calls[0]['name']}")
            elif isinstance(last_msg, ToolMessage):
                print(f"✅ Tool returned data: {last_msg.content}")
                
        print("------------------------------------\n")
        
        # After the stream finishes (reaches END), grab the final state
        final_state = app.get_state(config)
        final_message = final_state.values["messages"][-1]
        print(f"Assistant: {final_message.content}\n")

if __name__ == "__main__":
    main()
