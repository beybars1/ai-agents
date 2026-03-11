import os
from dotenv import load_dotenv
import autogen

load_dotenv()

# Specify the LLM configuration for AutoGen
llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]
}

def main():
    print("🤝 Welcome to Module 6: Multi-Agent Systems (AutoGen)!")
    print("Watch as multiple specialized agents collaborate to solve a problem.\n")

    # ---------------------------------------------------------
    # 1. Create the Agents (The Personas)
    # ---------------------------------------------------------
    
    # The UserProxyAgent acts as the "manager" or the proxy for the human user.
    # It initiates the chat and can optionally execute code or ask human for input.
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        system_message="A human admin.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        max_consecutive_auto_reply=10
    )

    # The first LLM agent: A specialized Researcher
    researcher = autogen.AssistantAgent(
        name="Researcher",
        llm_config=llm_config,
        system_message="""You are an expert researcher. 
        Your job is to gather accurate information, facts, and structure ideas.
        Do not write the final polished piece, just provide the raw data and outline.
        When you are done, just say 'I have finished gathering information' and nothing else."""
    )

    # The second LLM agent: A specialized Writer
    writer = autogen.AssistantAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="""You are an expert writer. 
        Your job is to take the raw data provided by the Researcher and write a beautiful, engaging short story or article.
        You should wait for the Researcher to provide the data first.
        When your writing is complete, reply with 'TERMINATE' to end the conversation."""
    )

    # ---------------------------------------------------------
    # 2. Setup the Group Chat
    # ---------------------------------------------------------
    
    # AutoGen uses a GroupChat object to manage the conversation between more than two agents.
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, writer],
        messages=[],
        max_round=12
    )

    # The GroupChatManager orchestrates who speaks next based on the conversation context
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # ---------------------------------------------------------
    # 3. Start the Conversation
    # ---------------------------------------------------------
    
    user_request = input("Enter a topic you want the team to write about (e.g., 'The history of coffee'):\n> ")
    if not user_request:
        user_request = "The history of coffee"
        
    print("\n--- Starting the multi-agent collaboration ---")
    
    # The proxy initiates the chat with the manager
    user_proxy.initiate_chat(
        manager,
        message=f"Team, please research and write a short, engaging article about this topic: {user_request}"
    )

if __name__ == "__main__":
    main()
