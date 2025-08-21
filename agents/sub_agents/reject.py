from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent


class RejectSubAgent(BaseSubAgent):
    """Fallback agent that politely declines unsupported tasks.

    Always receives a raw prompt string and wraps it as a PromptTemplate
    with {user_query}.
    """

    def __init__(self, model: str, prompt: str) -> None:
        self.name = "reject_agent"
        self.description = "Rejects unsupported tasks with a polite message."
        self._prompt = PromptTemplate(
            input_variables=["user_query"],
            template=prompt,
        )

        llm = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self._prompt,
            name="reject",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        last_user_msg = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
        out = self.agent.invoke({
            "messages": state["messages"],
            "user_query": last_user_msg,
        })
        return {"messages": out["messages"], "agent": self.name}
