# run_chat.py

from agents.manager_agent.manager_agent import ManagerAgent
from langchain_core.messages import AIMessage

def main():
    agent = ManagerAgent()

    print("🤖 Hello! Ask me a news-related question or request a summary.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        # Stateless: no history passed
        messages = agent.chat(message=user_input, history=[])

        # Print latest AI response
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"\n🤖 AI: {last_ai.content}\n")
        else:
            print("\n⚠️ No response.\n")

if __name__ == "__main__":
    main()
