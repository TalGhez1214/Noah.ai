from agents.sub_agents.qa import QASubAgent
from agents.sub_agents.summarizer import SummarySubAgent
from rag.rag_piplines.rag_retriever import RAGRetriever

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ManagerAgent:
    def __init__(self):
        # Create ONE retriever and share it
        self.retriever = RAGRetriever()

        deps = {"retriever": self.retriever}

        self.sub_agents = {
            "qa": QASubAgent(deps=deps),
            "summary": SummarySubAgent(deps=deps),
        }

        self.router_chain = LLMChain(
            llm=ChatOpenAI(temperature=0),
            prompt=PromptTemplate.from_template(
                "You are a router. Available agents:\n{agent_descriptions}\n"
                "Query: {query}\n"
                "Answer with only the agent key (e.g., 'qa', 'summary')."
            )
        )

    def route(self, query: str) -> str:
        descriptions = "\n".join(
            f"- {k}: {a.describe()}" for k, a in self.sub_agents.items()
        )
        key = self.router_chain.run(agent_descriptions=descriptions, query=query).strip()

        if key not in self.sub_agents:
            return f"🤖 Sorry, I couldn't understand which agent to use for: {query}"

        response = self.sub_agents[key].run(query)
        return f"🔁 Routed to `{key}` agent:\n\n{response}"
