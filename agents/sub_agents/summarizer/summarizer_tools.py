from typing import Any, Dict, Optional
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from pymongo import MongoClient
import re
import os

# ---- import search graph module ----
import rag.rag_piplines.articles_finder_graph as M

URL_RE = re.compile(r"https?://\S+")

def _extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = URL_RE.search(text)
    return m.group(0) if m else None

# MongoDB setup

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")

if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client[DB]
    mongo_articles_collection = db[COLL]
else:
    mongo_articles_collection = None

## build retriever
retriever = M.build_graph()

# ---------- SUMMARIZER TOOLS ----------

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

    if mongo_articles_collection is not None:
        try:
            doc = mongo_articles_collection.find_one({"url": url})
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

    initial = {
        "messages": state.get("messages", []),
        "user_query": state.get("user_query", "") or ""
    }
    out: Dict[str, Any] = retriever.invoke(initial)
    top_results = out.get("top_results", []) or []
    # Return just the best article (empty dict if nothing found)
    return top_results if top_results else {}