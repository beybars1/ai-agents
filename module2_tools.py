import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------
# 1. Define the actual Python functions (The "Tools")
# ---------------------------------------------------------
def get_weather(location: str) -> str:
    """Mock function to get the weather for a specific location."""
    # In a real app, you would call a real weather API here
    mock_db = {
        "tokyo": "Sunny and 75°F",
        "london": "Rainy and 55°F",
        "new york": "Cloudy and 65°F"
    }
    return mock_db.get(location.lower(), f"Weather data not found for {location}.")

def calculate_math(expression: str) -> str:
    """Mock function to calculate a simple math expression safely."""
    try:
        # DO NOT DO THIS IN PRODUCTION WITH UNTRUSTED INPUT
        # This is just a simple eval for demonstration purposes
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {e}"

# ---------------------------------------------------------
# 2. Describe the tools to the OpenAI model (JSON Schema)
# ---------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco or Tokyo",
                    }
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. 2 + 2 or 10 * 5",
                    }
                },
                "required": ["expression"],
            },
        }
    }
]

# ---------------------------------------------------------
# 3. Main Chat Loop
# ---------------------------------------------------------
def main():
    print("🛠️  Welcome to the Module 2 Tool-Calling Chatbot!")
    print("Try asking about the weather in London, or ask it to calculate 25 * 4.")
    print("Type 'exit' to quit.\n")

    messages = [{"role": "system", "content": "You are a helpful assistant with access to tools."}]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print(messages)
            break

        messages.append({"role": "user", "content": user_input})

        # Step 1: Call the model with tools enabled
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto" # Let the model decide if it needs to use a tool
        )
        
        message = response.choices[0].message
        
        # Step 2: Check if the model decided to use any tools
        if message.tool_calls:
            # We append the assistant message so the memory knows the model decided to call a tool
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name

                # Parse the JSON arguments the model provided
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"   [Agent is calling tool -> {function_name}({arguments})]")
                
                # Step 3: Execute our local Python function
                if function_name == "get_weather":
                    tool_result = get_weather(arguments.get("location"))
                elif function_name == "calculate_math":
                    tool_result = calculate_math(arguments.get("expression"))
                else:
                    tool_result = f"Error: Unknown function {function_name}"
                
                print(f"   [Tool Result -> {tool_result}]")

                # Step 4: Add the tool's output back to the message list
                # We need to pass the tool_call_id so the LLM knows which call this result is for
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            # Step 5: Call the model a SECOND time, now that it has the tool results
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            final_reply = second_response.choices[0].message.content
            print(f"\nAssistant: {final_reply}\n")
            messages.append({"role": "assistant", "content": final_reply})
            
        else:
            # The model didn't use a tool, it just replied normally
            print(f"\nAssistant: {message.content}\n")
            messages.append({"role": "assistant", "content": message.content})


if __name__ == "__main__":
    main()
