from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent


class FallbackSubAgent(BaseSubAgent):
    """Fallback agent that politely declines unsupported tasks.

    Always receives a raw prompt string and wraps it as a PromptTemplate
    with {user_query}.
    """

    def __init__(self, model: str, prompt: str) -> None:
        self.name = "fallback_agent"
        self.description = "Rejects unsupported tasks with a polite message."
        self._prompt = prompt

        llm = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self._prompt,
            name="fallback_agent",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
                                        
        out = self.agent.invoke(
                                {"messages": state["messages"]},  # still required even if unused
                                config={
                                    "configurable": {
                                        "user_query": state["user_query"],
                                    }
                                }
                            )
        return {"messages": out["messages"], "agent": self.name}
