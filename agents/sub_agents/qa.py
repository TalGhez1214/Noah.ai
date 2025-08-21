from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent


class QASubAgent(BaseSubAgent):
    def __init__(self, retriever, model: str, prompt: str, top_k: int = 6) -> None:
        self.name = "qa_agent"
        self.description = "Q&A agent that answers user questions using retrieved knowledge."
        self.retriever = retriever
        self.top_k = top_k
        self.prompt = prompt

        @tool("get_knowledge_for_answer")
        def _get_knowledge_for_answer(query: str) -> str:
            try:
                hits = self.retriever.retrieve(
                    question=query,
                    mode="chunk",
                    k_initial_matches=80,
                    k_final_matches=self.top_k,
                )
            except Exception:
                return ""
            chunks = []
            for h in hits:
                ch = (h.get("chunk") or "").strip()
                if ch:
                    chunks.append(ch)
            return "\n\n---\n\n".join(chunks)

        llm = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=llm,
            tools=[_get_knowledge_for_answer],
            prompt=self.prompt,
            name="qa",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return self.agent.tools[0].run(query) 

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.agent.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": self.name}
