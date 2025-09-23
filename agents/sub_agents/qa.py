from __future__ import annotations
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from .base import BaseSubAgent
from typing_extensions import Annotated
from langgraph.prebuilt import create_react_agent, InjectedState



class QASubAgent(BaseSubAgent):
    def __init__(
        self,
        retriever,
        model: str,
        prompt: str,
    ) -> None:
        self.name = "qa_agent"
        self.description = "Q&A agent that answers user questions using retrieved knowledge."
        self.retriever = retriever
        self.prompt = prompt

        # ---------- TOOLS ----------

        @tool(
            "get_data_for_answer_from_database_tool",
            description=(
                "Retrieve relevant data from the database to answer the user's question. "
            ),
        )
        def get_data_for_answer_from_database_tool(
            state: Annotated[dict, InjectedState],  # injected automatically by LangGraph
        ) -> dict:
            # 🔧 Force this agent’s default to 1 (must be done BEFORE build_graph)

            initial = {
                "messages": state.get("messages", []),
                "user_query": state.get("user_query", "") or "",
                "file_type": "chunks",
                "requested_k": 10, # Number of chunks to return
            }
            out: Dict[str, Any] = self.retriever.invoke(initial)
            top_results = out.get("top_results", []) or []
            # Return just the best article (empty dict if nothing found)
            return top_results if top_results else {}

        self._tools = [get_data_for_answer_from_database_tool]

        # ---------- LLM + Agent ----------

        llm = ChatOpenAI(model=model, temperature=0.8)
        
        self.agent = create_react_agent(
            model=llm,
            tools=self._tools,
            prompt=self.prompt,
            name="qa",
        )

    def get_knowledge_for_answer(self, query: str) -> str:        
        pass

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.agent.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": self.name}
