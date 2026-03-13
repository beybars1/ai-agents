import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

load_dotenv()

# ==========================================
# 1. The Agent Logic (From Module 5)
# ==========================================
# We define a basic LangGraph agent with a tool to simulate 
# a production backend service doing real work.

@tool
def get_user_data(user_id: str) -> str:
    """Mock database lookup tool."""
    return f"User {user_id} has account status: Active."

tools = [get_user_data]

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_node(state: State):
    messages = state["messages"]
    if len(messages) == 1:
        messages = [SystemMessage(content="You are a production customer service agent.")] + messages
    return {"messages": [llm.invoke(messages)]}

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

app_agent = graph_builder.compile()


# ==========================================
# 2. Production API (FastAPI)
# ==========================================
# In the real world, you don't use 'input()' loops in a terminal.
# You build an asynchronous API that frontend apps (React/Vue) can connect to.

app = FastAPI(title="Module 7: Production Agent API")

# A basic frontend HTML page to test our agent
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Chat</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; }
        #chat { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 20px;}
        .msg { margin: 5px 0; padding: 8px; border-radius: 4px; }
        .user { background: #e3f2fd; text-align: right; }
        .agent { background: #f5f5f5; }
        .tool { background: #fff3e0; font-size: 0.8em; font-family: monospace; }
        input[type="text"] { width: 80%; padding: 8px; }
        button { width: 15%; padding: 8px; }
    </style>
</head>
<body>
    <h2>Module 7: Production Agent</h2>
    <div id="chat"></div>
    <input type="text" id="userInput" placeholder="Ask something (e.g. check user ID 123)"/>
    <button onclick="sendMessage()">Send</button>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('userInput');

        function appendMessage(role, txt, className) {
            const div = document.createElement('div');
            div.className = 'msg ' + className;
            div.innerHTML = `<strong>${role}:</strong> ${txt}`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMessage() {
            const text = input.value;
            if(!text) return;
            appendMessage('You', text, 'user');
            input.value = '';
            
            // We connect to our backend's streaming endpoint via Server-Sent Events (SSE)
            const responseStream = new EventSource(`/chat_stream?message=${encodeURIComponent(text)}`);
            
            responseStream.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'tool') {
                    appendMessage('System', data.content, 'tool');
                } else if (data.type === 'agent') {
                    appendMessage('Agent', data.content, 'agent');
                    responseStream.close(); // Close stream when finished
                }
            };
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    """Serves the basic HTML testing interface."""
    return HTMLResponse(html_template)

@app.get("/chat_stream")
async def chat_stream(message: str):
    """
    An asynchronous streaming endpoint.
    It feeds the user's message into LangGraph, and 'yields' events 
    back to the frontend in real-time as the agent thinks and uses tools.
    """
    async def event_generator():
        # Notice we use `astream` because FastAPI is asynchronous
        async for event in app_agent.astream({"messages": [("user", message)]}):
            for node_name, state_update in event.items():
                last_msg = state_update["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    # Tell the frontend a tool is being used
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "tool", "content": f"[Executing Tool: {last_msg.tool_calls[0]['name']}]"})
                    }
                elif "tool" in node_name:
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "tool", "content": f"[Tool Result: {last_msg.content}]"})
                    }
                else: # The final agent response
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "agent", "content": last_msg.content})
                    }

    # Return a Server-Sent Events response
    import json
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import base64
    
    print("🚀 Module 7: Starting Production API Server")
    print("Running on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.\n")
    
    # Run the ASGI web server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
