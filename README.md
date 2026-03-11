# AI Agents: From 0 to Mastery

Welcome to the AI Agents Mastery learning path! This project contains a series of modules designed to take you from basic LLM API interactions all the way to building production-ready, multi-agent systems using Python.

We will focus primarily on **LangGraph** for building reliable, deterministic agent architectures, and will explore **AutoGen** for conversational multi-agent systems.

## Progress Tracker

- [ ] **Module 1: Foundations (The Brain)**
  - Actions: Build a basic terminal chatbot that maintains conversation history.
  - Concepts: API basics, System Prompts, Few-Shot Prompting, Context Windows.
- [ ] **Module 2: Tool Use & Functions (The Hands)**
  - Actions: Build an agent that can fetch real-time weather and calculate math equations.
  - Concepts: Function Calling, Structured Outputs.
- [ ] **Module 3: RAG & External Knowledge (The Bookshelf)**
  - Actions: Build an agent that can answer questions based on a specific PDF or local documentation.
  - Concepts: Embeddings, Vector Databases, Retrieval-Augmented Generation.
- [ ] **Module 4: Agentic Loops & Frameworks (The Coordinator)**
  - Actions: Build an agentic loop from scratch, then migrate it to LangGraph.
  - Concepts: The ReAct pattern, state graph frameworks.
- [ ] **Module 5: Memory & Planning (The Strategist)**
  - Actions: Build a coding agent that formulates a plan, writes code, reviews it, and fixes errors using LangGraph.
  - Concepts: Short-term vs. Long-term memory, semantic memory, Plan-and-Solve patterns, Reflection/Self-Correction.
- [ ] **Module 6: Multi-Agent Systems (The Team)**
  - Actions: Build a simulated company (e.g., Researcher + Writer + Editor) collaborating on a report using AutoGen.
  - Concepts: Agent collaboration, roles, delegation, hierarchical graphs.
- [ ] **Module 7: Production & Evaluation (The Professional)**
  - Actions: Deploy an agent as a robust API with LangSmith observability.
  - Concepts: Streaming tool calls, Observability, Evaluation rubrics, jailbreak/prompt injection defenses.

## Getting Started

### Prerequisites
1. Ensure you have Python installed.
2. Get an OpenAI API Key from [platform.openai.com](https://platform.openai.com/).

### Running Module 1: The Basic Chatbot
Module 1 introduces the fundamental concepts of talking to an LLM: sending a "System Prompt" and maintaining a history of messages so the model remembers the context of the conversation.

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install openai python-dotenv
   ```
2. Open the `.env` file and add your API key:
   ```
   OPENAI_API_KEY="sk-..."
   ```
3. Run the chatbot:
   ```bash
   python module1_chatbot.py
   ```

### Running Module 2: Tool Use & Functions
Module 2 teaches the model how to interact with the outside world by providing it with a list of "tools" (Python functions) it can conditionally choose to call, such as checking the weather or doing math.

1. Ensure your virtual environment is active and `.env` is configured.
2. Run the tool-calling bot:
   ```bash
   python module2_tools.py
   ```
   *Try asking it "What's the weather in Tokyo?" or "What is 45 times 12?"*

### Running Module 3: RAG & External Knowledge
Module 3 demonstrates Retrieval-Augmented Generation (RAG) entirely from scratch using `numpy` and OpenAI embeddings. You will see exactly how we convert text to vectors (arrays of numbers), measure semantic similarities, and retrieve relevant chunks of data to give the model context it wasn't originally trained on.

1. Ensure your virtual environment is active.
2. We need `numpy` to calculate similarities. Install it if you haven't:
   ```bash
   pip install numpy
   ```
3. Run the RAG bot:
   ```bash
   python module3_rag.py
   ```
   *Try asking it "Who is the boss of the club?" or "What is your favorite snack?" based on the hidden data in `dummy_data.txt`.*

### Running Module 4: Agentic Loops & Frameworks (LangGraph)
Module 4 introduces **LangGraph**, the framework we will use for the rest of the course to build robust agent architectures. It demonstrates the concept of a "StateGraph" where a shared "State" object (in this case, our message history) is passed between "Nodes" (functions or LLMs).

1. Ensure your virtual environment is active.
2. Install the necessary LangGraph and LangChain packages:
   ```bash
   pip install langgraph langchain-openai langchain-core
   ```
3. Run the LangGraph bot:
   ```bash
   python module4_langgraph.py
   ```
   *Notice how the terminal output now explicitly shows `[Node executed: chatbot]` before answering. The flow of data is explicitly controlled by the graph.*

### Running Module 5: Memory & Planning (The Strategist)
Module 5 combines everything we've learned so far! We use LangGraph to build a true **Agentic Loop** with tools and persistent memory.

Unlike Module 4, this agent has a "ToolNode" added to its graph, and a conditional edge. If the LLM decides it needs to use a tool, the graph pauses the LLM, runs the tool, appends the result to the memory, and then loops *back* to the LLM so it can read the result and continue. 

1. Ensure your virtual environment is active.
2. Run the Agentic Loop:
   ```bash
   python module5_agentic_loop.py
   ```
   *Try asking it a multi-stage goal: "What is the weather in London, and what is the temperature in Fahrenheit minus 10?" Watch its internal thought process as it loops through multiple tool calls autonomously!*

### Running Module 6: Multi-Agent Systems (AutoGen)
Module 6 shifts gears from LangGraph to **AutoGen** to demonstrate how to build Multi-Agent Systems. Instead of one agent doing everything, we create specialized "Personas" (a Researcher and a Writer) and place them in a Group Chat. They will autonomously converse with each other to complete a goal.

1. Ensure your virtual environment is active.
2. Install AutoGen:
   ```bash
   pip install pyautogen
   ```
3. Run the AutoGen script:
   ```bash
   python module6_autogen.py
   ```
   *Provide a topic, sit back, and watch the agents collaborate in real-time!*
