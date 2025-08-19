from typing import TypedDict, Annotated, Literal
import operator
import re

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# This allow us to save memory between different API calls and sessions - 
# This is in-memory only - It does not persist between server restarts, Good for dev/testing, not production
memory = MemorySaver() 

from agents.prompts import SUPERVISOR_PROMPT
from agents.sub_agents.qa import build_qa_agent
from agents.sub_agents.summarizer import build_summary_agent
from agents.sub_agents.articles_finder_agent import ArticalFinderAgent
from rag.rag_piplines.rag_retriever import RAGRetriever
from typing import Optional


class AgentState(MessagesState):
    agent: Optional[str] = None


class ManagerAgent:
    def __init__(self, model: str = "gpt-4o-mini", user_query: str = "", user_id: Optional[str] = None):
        self.user_query = user_query
        self.user_id = user_id
        self.retriever = RAGRetriever()
        self.qa_app = build_qa_agent(self.retriever, model)
        self.summary_app = build_summary_agent(self.retriever, model)
        self.articles_finder_agent = ArticalFinderAgent(retriever=self.retriever, model=model).build_articles_finder_agent()
        self.router_llm = ChatOpenAI(model=model, temperature=0)

        # LangGraph build
        graph = StateGraph(AgentState)
        graph.add_node("qa", self._qa_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("articles_finder", self._articles_finder_node)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("reject", self._reject_node)

        graph.add_edge(START, "supervisor")
        graph.add_edge("qa", END)
        graph.add_edge("summary", END)
        graph.add_edge("reject", END)

        self.app = graph.compile(checkpointer=memory)

    def _qa_node(self, state: AgentState) -> AgentState:
        out = self.qa_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "qa"}

    def _summary_node(self, state: AgentState) -> AgentState:
        out = self.summary_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "summary"}
    
    def _articles_finder_node(self, state: AgentState) -> AgentState:
        """
        This node handles the articles finder agent, which retrieves relevant articles
        based on the user's query and returns them as a dictionary response for each article:
        {
            "title": "Article Title",
            "Summary": "Article Summary",
            "Key Quote": "Key Quote from the article",
        }
        """
        articles = self.articles_finder_agent.get_knmowledge_for_answer(user_query=self.user_query)
        articles_snippets = []

        for article in articles:
            agent_answer = self.articles_finder_agent.build_articles_finder_agent(user_query=self.user_query, article=article).invoke({"messages": state["messages"]})
            try:
                json_output = self.articles_finder_agent.structured_output(agent_answer)
            except Exception as e:
                print(f"Error parsing structured output: {e}")
                json_output = {"Summary": "No summary available", "Key Quote": "No quote available"}
            articles_snippets.update(json_output)

        for i in range(len(articles_snippets)):
            articles_snippets[i]["title"] = articles[i]["title"]
            articles_snippets[i]["url"] = articles[i]["url"] # Make sure it's clickable in the UI

        return {
            "messages": [AIMessage(content=articles_snippets)], 
            "agent": "articles_finder"
        }
    

    def _reject_node(self, state: AgentState) -> AgentState:
        """Politely decline unsupported tasks using an LLM-generated message."""
        last_user_msg = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
        
        rejection_prompt = (
            "You are a helpful assistant specialized in news Q&A and summarization.\n"
            "A user has just asked a question or made a request that is outside your capabilities.\n"
            "You cannot help with this request, but you want to be kind and clear.\n\n"
            "Explain that you are limited to:\n"
            "- Answering news-related questions\n"
            "- Summarizing articles or topics\n"
            "Then give 1–2 example prompts the user *can* ask instead.\n\n"
            "User said:\n"
            f"{last_user_msg}\n\n"
            "Respond kindly and clearly:"
        )

        reply = self.router_llm.invoke([SystemMessage(content=rejection_prompt)]).content.strip()
        return {"messages": [AIMessage(content=reply)], "agent": "none"}

    def _supervisor(self, state: AgentState) -> Command:
        last_user = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
        msgs = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=last_user)]
        choice = self.router_llm.invoke(msgs).content.strip().lower()

        if "summary" in choice:
            return Command(goto="summary")
        elif "qa" in choice:
            return Command(goto="qa")
        elif "articles_finder" in choice:
            return Command(goto="articles_finder")
        elif "finish" in choice:
            return Command(goto="reject")
        else:
            # Fallback: use question mark as a hint
            if re.search(r"\?$", last_user.strip()):
                return Command(goto="qa")
            return Command(goto="reject")

    def chat(self) -> list[BaseMessage]:
        """Chat entry point for full conversation."""
        thread = {"configurable": {"thread_id": self.user_id}} # Use user_id as thread_id to pull conversation history from RAM
        new_message = [HumanMessage(content=self.user_query)]

        updated = self.app.invoke({"messages": new_message}, thread) # calling manager agent to start working

        for m in updated["messages"]:
            m.pretty_print()
        return updated["messages"]
