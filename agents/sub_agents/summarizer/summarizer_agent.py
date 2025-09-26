from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState

from ..base import BaseSubAgent
from typing_extensions import Annotated
from agents.manager_agent.GraphStates import ReactAgentState
from agents.sub_agents.summarizer.summarizer_tools import (
    summary_content_from_link_tool,
    summary_article_from_current_page_tool,
    get_articles_from_database_tool,
)

# ---------- Structured Output ----------
class RespondFormat(BaseModel):
    title: str = Field(..., description="The article's title")
    summary: str = Field(..., description="2–5 sentence summary of the article")
    url: str = Field(..., description="Canonical article URL, empty string if unknown")


class SummarizerSubAgent(BaseSubAgent):
    def __init__(self, retriever, model: str, prompt: str) -> None:
        self.name = "summary_agent"
        self.description = "This agent is responsible for all article summarizing requests."
        self.retriever = retriever
        self.prompt = prompt

        # ---------- TOOLS ----------
        self.tools = [
            summary_content_from_link_tool,
            summary_article_from_current_page_tool,
            get_articles_from_database_tool,
        ]

        # ---------- LLMs ----------
        self.llm = ChatOpenAI(model=model, temperature=0.2)
        self.tool_enabled_llm = self.llm.bind_tools(self.tools)
        self.parser_model = self.llm.with_structured_output(RespondFormat)

        # ---------- Build Graph ----------
        workflow = StateGraph(ReactAgentState)

        # agent node
        def call_model(state: ReactAgentState):
            response = self.tool_enabled_llm.invoke(state["messages"])
            return {"messages": [response]}

        # respond node (final structured answer)
        def respond(state: ReactAgentState):
            """
            Parse the LAST AI message (the model's final text after all tools),
            which you've guided to contain title, summary, url.
            """
            final_ai = state["messages"][-1]           # <- last AIMessage (no tool_calls)
            response = self.parser_model.invoke(
                [HumanMessage(content=final_ai.content)]
            )
            return {"final_response": response}

        # routing logic
        def should_continue(state: ReactAgentState):
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                return "respond" # no tool calls → we're done; route to respond, where we run the structured-output model
            return "continue" # there ARE tool calls → route to the tools node so the ToolNode will execute them

        # build workflow
        workflow.add_node("agent", call_model)
        workflow.add_node("respond", respond)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools", # Go to executing tools
                "respond": "respond", # Go to structured-output model - finish workflow
            },
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        self.graph = workflow.compile()

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.graph.invoke(state)
        return {
            "messages": out.get("messages", []),
            "agent": self.name,
            "final_json": (
                out["final_response"].model_dump()
                if "final_response" in out
                else {}
            ),
        }
