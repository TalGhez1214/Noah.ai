from typing import TypedDict, Annotated, Literal
import operator
import re

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

from agents.prompts import SUPERVISOR_PROMPT
from agents.sub_agents.qa import build_qa_agent
from agents.sub_agents.summarizer import build_summary_agent
from rag.rag_piplines.rag_retriever import RAGRetriever


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    agent: str


class ManagerAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.retriever = RAGRetriever()
        self.qa_app = build_qa_agent(self.retriever, model)
        self.summary_app = build_summary_agent(self.retriever, model)
        self.router_llm = ChatOpenAI(model=model, temperature=0)

        # LangGraph build
        graph = StateGraph(AgentState)
        graph.add_node("qa", self._qa_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("reject", self._reject_node)

        graph.add_edge(START, "supervisor")
        graph.add_edge("qa", END)
        graph.add_edge("summary", END)
        graph.add_edge("reject", END)

        self.app = graph.compile()

    def _qa_node(self, state: AgentState) -> AgentState:
        out = self.qa_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "qa"}

    def _summary_node(self, state: AgentState) -> AgentState:
        out = self.summary_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "summary"}

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
        elif "finish" in choice:
            return Command(goto="reject")
        else:
            # Fallback: use question mark as a hint
            if re.search(r"\?$", last_user.strip()):
                return Command(goto="qa")
            return Command(goto="reject")

    def chat(self, message: str, history: list[BaseMessage] = None) -> list[BaseMessage]:
        """Chat entry point for full conversation."""
        history = history or []
        full_state = {"messages": history + [HumanMessage(content=message)]}
        updated = self.app.invoke(full_state)
        for m in updated["messages"]:
            m.pretty_print()
        return updated["messages"]
