from agents.sub_agents.qa import QASubAgent
from agents.sub_agents.summarizer import SummarySubAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ManagerAgent:
    def __init__(self):
        self.sub_agents = {
            "qa": QASubAgent(),
            "summary": SummarySubAgent(),
        }

        # Used to decide which agent to route to
        self.router_chain = LLMChain(
            llm=ChatOpenAI(temperature=0),
            prompt=PromptTemplate.from_template(
                "You are a router. Decide which agent to use for this query.\n"
                "Available agents:\n{agent_descriptions}\n"
                "Query: {query}\n"
                "Answer with only the agent key (e.g., 'qa', 'summary')."
            )
        )

    def route(self, query: str) -> str:
        descriptions = "\n".join(
            [f"- {key}: {agent.describe()}" for key, agent in self.sub_agents.items()]
        )
        agent_key = self.router_chain.run(agent_descriptions=descriptions, query=query).strip()

        if agent_key not in self.sub_agents:
            return f"🤖 Sorry, I couldn't understand which agent to use for: {query}"

        response = self.sub_agents[agent_key].run(query)
        return f"🔁 Routed to `{agent_key}` agent:\n\n{response}"
