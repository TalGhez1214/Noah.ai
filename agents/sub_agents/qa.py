from __future__ import annotations
from typing import Any, Dict, List
from datetime import date
import json

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, InjectedState
from typing_extensions import Annotated

from langchain_tavily import TavilySearch

# ---------- Parser schema (Pydantic) ----------
from pydantic import BaseModel, Field
from typing import List as _List


class ResourcePickerSchema(BaseModel):
    """
    Structured output for the parser agent.
    Select up to 3 relevant DB chunk IDs that best support the user's question and the draft answer.
    """
    resource_list: _List[str] = Field(
        default_factory=list,
        description="Up to 3 exact chunk IDs from get_data_for_answer_from_database_tool that truly support the answer. [] if none."
    )


class QASubAgent:
    """
    Simple Q&A agent + end-of-run parser:

    Flow:
      1) ReAct QA agent (can call DB tool and/or web_search) → produces a draft answer (free-form text).
      2) Parser agent (with_structured_output(ResourcePickerSchema)) → reads the draft + tool trace and returns
         up to 3 relevant DB chunk IDs (or []).
      3) Final output:
         - messages[-1].content = the original draft answer (unchanged),
         - relevant_articles_for_user = selected IDs.
    """

    def __init__(self, retriever, model: str, prompt) -> None:
        self.name = "qa_agent"
        self.description = "This agent is responsible for all the question answering requests."
        self.retriever = retriever
        self.prompt = prompt
        self.model = model

        # ----- Tools -----
        
        web_search_tool = TavilySearch(
            max_results=5,
            name="web_search",
            description="Search the web for up-to-date information.",
        )

        self._tools = [web_search_tool]

        # ----- Base LLM + ReAct Agent -----
        self._llm = ChatOpenAI(model=self.model, temperature=0.2)
        #self._llm = ChatGroq(model=model, temperature=0.2)
        self.agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self.prompt,
            name="qa",
        )


    # ---------- Public API ----------

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        page_content = (state.get("current_page") or {}).get("content", "") or ""
        messages = state.get("messages", []) or []
        user_query = state.get("user_query", "") or ""

        # 1) Run the main ReAct QA agent (free to call tools as needed)
        agent_answer = self.agent.invoke(
            {"messages": messages},
            config={
                "configurable": {
                    "current_page_content": page_content,
                    "today": date.today().isoformat(),
                    "user_query": user_query,
                }
            },
        )

        return {
            "messages": agent_answer["messages"],
            "agent": self.name,
        }
