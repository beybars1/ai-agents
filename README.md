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
