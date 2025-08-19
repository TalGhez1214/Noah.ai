# run_chat.py

from agents.manager_agent.manager_agent import ManagerAgent
from langchain_core.messages import AIMessage

def main():
    
    user_id = "123" # TODO: Replace with actual user ID if needed using MongoDB request
    agent = ManagerAgent(user_id=user_id)

    print("🤖 Hello! Ask me a news-related question or request a summary.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        agent.user_query = input("🧑 You: ").strip()
        if agent.user_query.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        # Stateless: no history passed
        messages = agent.chat()
        

        # Print latest AI response
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"\n🤖 AI: {last_ai.content}\n")
        else:
            print("\n⚠️ No response.\n")

if __name__ == "__main__":
    main()
