import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env file
load_dotenv()

# Initialize the OpenAI client
# Ensure your .env file has OPENAI_API_KEY=your_key_here
client = OpenAI()

def main():
    print("🤖 Welcome to the Module 1 Basic Chatbot!")
    print("Type 'exit' to quit.\n")

    # The 'messages' list acts as the memory for the conversation.
    # We initialize it with a System Prompt.
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]

    while True:
        # 1. Get user input
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # 2. Add user message to memory
        messages.append({"role": "user", "content": user_input})

        try:
            # 3. Call the LLM API
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Using a fast, cost-effective model
                messages=messages
            )
            
            # 4. Extract the response text
            assistant_reply = response.choices[0].message.content
            print(f"\nAssistant: {assistant_reply}\n")
            
            # 5. Add assistant reply to memory so it remembers next time
            messages.append({"role": "assistant", "content": assistant_reply})
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Did you remember to set your OPENAI_API_KEY in the .env file?\n")

if __name__ == "__main__":
    main()
