from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent


class QASubAgent(BaseSubAgent):
    def __init__(
        self,
        retriever,
        prompt: str,
    ) -> None:
        self.name = "qa_agent"
        self.description = "Q&A agent that answers user questions using retrieved knowledge."
        self.retriever = retriever
        self.prompt = prompt

        @tool("get_knowledge_for_answer", description="Get knowledge for answer")
        def get_knowledge_tool(query: str) -> str:
            try:
                hits = self.retriever.retrieve(
                    query=query,
                    semantic_file="chunk",
                    k_semantic_matches=20,
                    k_final_matches=self.top_k,
                )
            except Exception:
                return ""
            chunks = [(h.get("chunk") or "").strip() for h in hits]
            return "\n\n---\n\n".join([ch for ch in chunks if ch])
        
        # Save the raw callable for testing
        self._knowledge_tool = get_knowledge_tool

        llm = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=llm,
            tools=[get_knowledge_tool],
            prompt=self.prompt,
            name="qa",
        )

    def get_knowledge_for_answer(self, query: str) -> str:        
        return self._knowledge_tool.run(query) 

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.agent.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": self.name}
