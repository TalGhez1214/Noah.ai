# agents/manager_agent/manager_agent.py
from typing import TypedDict, Annotated, Literal
import operator
import re

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

from agents.prompts import SUPERVISOR_PROMPT
from agents.sub_agents.qa import build_qa_agent
from agents.sub_agents.summarizer import build_summary_agent

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    agent: str

class ManagerAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.qa_app = build_qa_agent(model)
        self.summary_app = build_summary_agent(model)
        self.router_llm = ChatOpenAI(model=model, temperature=0)

        graph = StateGraph(AgentState)
        graph.add_node("qa", self._qa_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("supervisor", self._supervisor)

        graph.add_edge(START, "supervisor")
        graph.add_edge("qa", "supervisor")
        graph.add_edge("summary", "supervisor")

        self.app = graph.compile()

    def _qa_node(self, state: AgentState) -> AgentState:
        out = self.qa_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "qa"}

    def _summary_node(self, state: AgentState) -> AgentState:
        out = self.summary_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "summary"}

    def _supervisor(self, state: AgentState) -> Command[Literal["qa", "summary"]]:
        if state.get("agent"):
            return Command(goto=END)

        last_user = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
        msgs = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=last_user)]
        choice = self.router_llm.invoke(msgs).content.strip().lower()

        if "summary" in choice:
            goto = "summary"
        elif "qa" in choice:
            goto = "qa"
        elif "finish" in choice:
            return Command(goto=END)
        else:
            goto = "qa" if re.search(r"\?$", last_user.strip()) else "summary"

        return Command(goto=goto)

    def route(self, text: str) -> str:
        final = self.app.invoke({"messages": [HumanMessage(content=text)]})
        last_ai = next((m.content for m in reversed(final["messages"]) if m.type == "ai"), "")
        agent_used = final.get("agent", "unknown")
        return f"🔁 Routed to `{agent_used}` agent:\n\n{last_ai}"
