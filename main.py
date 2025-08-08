from agents.manager_agent.manager_agent import ManagerAgent
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    agent = ManagerAgent()
    
    while True:
        query = input("🧑 You: ")
        if query.lower() in ("exit", "quit"):
            break
        print("🤖 Agent:", agent.route(query))
