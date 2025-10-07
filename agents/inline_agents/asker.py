from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import date

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from agents.inline_agents.prompts import asker_prompt

class AskerAgent:
    """
    Stand-alone Asker agent (no Manager routing).
    - Receives page content + highlighted text + user's free-form question.
    - May use Tavily web_search for clarifications/updates.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.tools = [
            TavilySearchResults(
                max_results=5,
                name="web_search",
                description="Web search for quick definitions, updates, or context when the page is insufficient."
            )
        ]
        self.app = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=asker_prompt,
            name="asker",
        )

    def call(self, *, user_query: str, highlighted_text: str, current_page_content: str,
            thread_id: Optional[str] = None, ) -> List[BaseMessage]:
        # human message mostly for conversational flavor (prompt carries the grounding)
        human = HumanMessage(content=user_query or "Ask about the highlighted selection in context.")
        state: Dict[str, Any] = {"messages": [human]}

        result = self.app.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id or "inline-asker",
                    "current_page_content": current_page_content or "",
                    "highlighted_text": highlighted_text or "",
                    "user_query": user_query or "",
                }
            },
        )
        return result["messages"]
