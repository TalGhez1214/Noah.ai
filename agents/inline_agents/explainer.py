# agents/inline_agents/explainer.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import date

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage

# Prefer the community Tavily tool (most common as of 2025):
from langchain_community.tools.tavily_search import TavilySearchResults

# If your project already uses `langchain_tavily`, swap the import to:
# from langchain_tavily import TavilySearch as TavilySearchResults

from agents.inline_agents.prompts import explainer_prompt
from agents.prompts import qa_prompt


class ExplainerAgent:
    """
    Stand-alone Explainer agent (no Manager routing).
    - Prefers page content + highlighted text.
    - May use Tavily web_search for definitions/updates.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model_name = model
        self.temperature = temperature

        # LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # Tools (Tavily web search)
        self.tools = [
            TavilySearchResults(
                max_results=5,
                name="web_search",
                description="Web search for quick definitions or context when the page content is insufficient."
            )
        ]

        # ReAct agent
        self.app = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=explainer_prompt,  # from agents/inline_agents/propmts.py
            name="explainer",
        )

    def call(self, highlighted_text: str, current_page_content: str, user_query_hint: Optional[str] = None, 
             thread_id: Optional[str] = None) -> List[BaseMessage]:
        """
        Invoke the agent directly. Pass thread_id if you want per-user traces.
        """
        human = HumanMessage(content=user_query_hint or "Explain the highlighted selection in context.")
        state = {"messages": [human]}

        result = self.app.invoke(
            state,
            config={
                "configurable": {
                    "current_page_content": current_page_content or "",
                    "highlighted_text": highlighted_text or ""
                }
            },
        )
        return result["messages"]