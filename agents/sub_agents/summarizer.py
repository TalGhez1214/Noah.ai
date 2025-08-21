from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent


class SummarizerSubAgent(BaseSubAgent):
    def __init__(self, retriever, model: str, prompt: str, top_k: int = 5) -> None:
        self.name = "summary_agent"
        self.description = "Summarizes articles based on a given topic."
        self.retriever = retriever
        self.top_k = top_k
        self.prompt = prompt

        @tool("get_articles_for_summary")
        def _get_articles_for_summary(topic: str) -> str:
            try:
                hits = self.retriever.retrieve(
                    question=topic,
                    mode="article",
                    k_initial_matches=50,
                    k_final_matches=self.top_k,
                )
            except Exception:
                return ""
            contents = []
            for h in hits:
                c = (h.get("content") or "").strip()
                if c:
                    contents.append(c)
            return "\n\n---\n\n".join(contents)

        llm = ChatOpenAI(model=model, temperature=0.3)
        self.agent = create_react_agent(
            model=llm,
            tools=[_get_articles_for_summary],
            prompt=self.prompt,
            name="summary",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return self.agent.tools[0].run(query)

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.agent.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": self.name}
