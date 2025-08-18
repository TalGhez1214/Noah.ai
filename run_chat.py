# run_chat.py

from agents.manager_agent.manager_agent import ManagerAgent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def main():
    agent = ManagerAgent()
    history: list[BaseMessage] = []

    print("🤖 Hello! Ask me a news-related question or request a summary.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        history = agent.chat(message=user_input, history=history)
        # Get the last assistant reply
        last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)

        if last_ai:
            print(f"\n🤖 AI: {last_ai.content}\n")
        else:
            print("\n⚠️ No response.\n")

if __name__ == "__main__":
    main()
