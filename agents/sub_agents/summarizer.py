from __future__ import annotations
from typing import Any, Dict, Optional
import re
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, InjectedState
from .base import BaseSubAgent
from typing_extensions import Annotated
from agents.manager_agent.GraphStates import ReactAgentState

# ---- import search graph module ----
import rag.rag_piplines.articles_finder_graph as M

URL_RE = re.compile(r"https?://\S+")

def _extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = URL_RE.search(text)
    return m.group(0) if m else None

# ---------- Structured input for the DB/title tool ----------
class ArticleLookupSpec(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Exact or near-exact title of the article, if known."
    )
    article_content: Optional[str] = Field(
        default=None,
        description="any relevant lines and content from the artical or if avaiable a description the user provided."
    )
    author: Optional[str] = Field(
        default=None,
        description="Author's name, if known."
    )


class SummarizerSubAgent(BaseSubAgent):
    def __init__(
        self,
        retriever,
        model: str,
        prompt: str,
        mongo_articles_collection=None,  # pymongo.Collection (optional but recommended)
    ) -> None:
        self.name = "summary_agent"
        self.description = "This agent responsible for all the article summarizing requests."
        self.retriever = retriever
        self.prompt = prompt
        self.mongo_articles_collection = mongo_articles_collection 

         # ---------- TOOLS ----------

        @tool(
            "summary_content_from_link_tool",
            description=(
                "Fetch FULL article content for a given URL. "
            ),
        )
        def summary_content_from_link_tool(link_or_message: str) -> str:
            """
            Inside the tool:
              1) Extract the first URL from the input (or treat input as URL).
              2) Query MongoDB: articles_collection.find_one({'url': url})
              3) Return the 'content' field (plain text). If not found, return '' (no fallback).
            """
            url = _extract_first_url(link_or_message) or (link_or_message or "").strip()
            if not url or not url.startswith("http"):
                return ""

            if self.mongo_articles_collection is not None:
                try:
                    doc = self.mongo_articles_collection.find_one({"url": url})
                    return doc
                except Exception:
                    # Silent failure → return empty so the policy can ask for clarification.
                    pass

            # Explicitly NO retriever fallback by design.
            return "No content found from the provided URL."
        
        @tool(
            "summary_article_from_current_page_tool",
            description=(
                "Fetch FULL article content from the user's CURRENT page/tab. "
            ),
        )
        def summary_article_from_current_page_tool(
            state: Annotated[dict, InjectedState]  # injected automatically by LangGraph
        ) -> str:
            try:
                doc = state.get("current_page", {}) or {}
                return doc
            except Exception:
                return {}
            
        @tool(
            "get_articles_from_database_tool",
            description=(
                "Retrieve a single article document from the database using the Search Graph. "
                "Returns the top ranked article (title, author, content, url, etc.), or {} if none."
            ),
        )
        def get_articles_from_database_tool(
            state: Annotated[dict, InjectedState],  # injected automatically by LangGraph
        ) -> dict:
            # 🔧 Force this agent’s default to 1 (must be done BEFORE build_graph)
            M.REQUESTED_K_DEFAULT = 1
            # Compile the graph and run it with whatever the agent already has
            app = M.build_graph()

            initial = {
                "messages": state.get("messages", []),
                "user_query": state.get("user_query", "") or ""
            }
            out: Dict[str, Any] = app.invoke(initial)
            top_results = out.get("top_results", []) or []
            # Return just the best article (empty dict if nothing found)
            return top_results if top_results else {}
            
        self.tools = [
                        summary_content_from_link_tool,
                        summary_article_from_current_page_tool,
                        get_articles_from_database_tool,  
                    ]
            
        # ---------- LLM + Agent ----------
        
        llm = ChatOpenAI(model=model, temperature=0.2)

        self.agent = create_react_agent(
            model=llm,
            tools=self.tools,
            prompt=self.prompt,
            name="summary",
            state_schema=ReactAgentState
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = self.agent.invoke(state)
        return {"messages": out["messages"], "agent": self.name}
    
