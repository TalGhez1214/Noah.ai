# tests/test_summarizer_agent.py
import os
import re
import json
import uuid
import pytest

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agents.sub_agents.summarizer.summarizer_agent import SummarizerSubAgent
from agents.prompts import SUMMARY_PROMPT
import rag.rag_piplines.articles_finder_graph as M

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")


# ---------- helpers ----------
def _tool_map(agent):
    return {t.name: t for t in getattr(agent, "tools", [])}

def _tool_call(tool, *args, **kwargs):
    if hasattr(tool, "invoke"):
        return tool.invoke(*args, **kwargs)
    if hasattr(tool, "run"):
        return tool.run(*args, **kwargs)
    if hasattr(tool, "func"):
        return tool.func(*args, **kwargs)
    raise AttributeError("Unsupported tool interface (expected .invoke/.run/.func)")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


@pytest.fixture
def summarizer_agent():
    return SummarizerSubAgent(
        retriever=M.build_graph(),
        model="gpt-4o-mini",
        prompt=SUMMARY_PROMPT,
    )

@pytest.fixture
def article_doc():
    article_content = (
        "Yisrael Einhorn, a former campaign adviser to ex-prime minister Benjamin Netanyahu, "
        "crafted pro-Qatar messages and sent them to journalists Urich and Feldstein for publication. "
        "Einhorn's communications were part of a broader effort to influence public opinion in favor of Qatar, "
        "which included drafting articles and coordinating with media outlets. "
        "The revelations have raised questions about the ethical implications of such activities "
        "and the role of political operatives in shaping media narratives."
        "It was reported that Einhorn had been in contact with Urich and Feldstein, and that they had published some of Einhorn's messages."
    )
    return {
        "_id": {"$oid": "6895c5ccc895b660c6fb3ea0"},
        "url": "https://www.timesofisrael.com/ex-pm-aide-crafted-pro-qatar-messages-sent-to-urich-feldstein-to-publish-in-media/",
        "author": ["Ethan Rubinson"],
        "content": article_content,
        "fetched_at": "2025-08-08T09:39:23.425560+00:00",
        "published_at": "2025-08-09T16:45:00Z",
        "section": None,
        "source": "www.timesofisrael.com",
        "title": "Ex-PM aide crafted pro-Qatar messages, sent to Urich, Feldstein to publish in media",
    }


# ---------- tests ----------
def test_summarizer_agent_init(summarizer_agent):
    assert summarizer_agent.name == "summary_agent"
    assert hasattr(summarizer_agent, "graph")
    tm = _tool_map(summarizer_agent)
    assert "summary_content_from_link_tool" in tm
    assert "summary_article_from_current_page_tool" in tm
    assert "get_articles_from_database_tool" in tm or "get_articles_for_summary_from_database_tool" in tm


def test_link_tool_returns_content_from_mongo(summarizer_agent, article_doc):
    tm = _tool_map(summarizer_agent)
    tool = tm["summary_content_from_link_tool"]
    out = _tool_call(tool, f"please summarize {article_doc['url']}")
    assert isinstance(out, dict)
    assert article_doc["title"] == out.get("title")


def test_current_page_tool_reads_injected_state(summarizer_agent, article_doc):
    tm = _tool_map(summarizer_agent)
    tool = tm["summary_article_from_current_page_tool"]
    state = {"current_page": article_doc}
    out = tool.func(state) if hasattr(tool, "func") else _tool_call(tool, state)
    assert isinstance(out, dict)
    assert "Yisrael Einhorn, a former campaign adviser" in out["content"]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_modal_from_link_route_has_required_fields(summarizer_agent, article_doc):
    state = {"messages": [HumanMessage(content=f"Can you summarize {article_doc['url']} ?")]}
    result = summarizer_agent.graph.invoke(state)

    # messages sanity
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)

    # modal payload checks
    assert "modals" in result and isinstance(result["modals"], list) and result["modals"]
    payload = result["modals"][-1]
    assert payload.get("type") == "summary"
    for k in ("answer", "title", "url", "summary"):
        assert k in payload and isinstance(payload[k], str) and payload[k].strip()

    # title/url match
    #assert _normalize(payload["title"]) == _normalize(article_doc["title"])
    assert payload["url"].strip() == article_doc["url"].strip()
    assert _normalize(payload["answer"]) == _normalize(result["messages"][-1].content)

    for m in result["messages"]:
        m.pretty_print()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_modal_from_current_page_has_required_fields(summarizer_agent, article_doc):
    state = {
        "messages": [HumanMessage(content="Can you send me a summary of this article?.")],
        "current_page": article_doc,
    }
    result = summarizer_agent.graph.invoke(state)

    assert "messages" in result and isinstance(result["messages"][-1], AIMessage)
    assert "modals" in result and isinstance(result["modals"], list) and result["modals"]
    payload = result["modals"][-1]
    assert payload.get("type") == "summary"
    for k in ("answer", "title", "url", "summary"):
        assert k in payload and isinstance(payload[k], str) and payload[k].strip()

    # title/url match
    assert _normalize(payload["title"]) == _normalize(article_doc["title"])
    assert payload["url"].strip() == article_doc["url"].strip()
    assert _normalize(payload["answer"]) == _normalize(result["messages"][-1].content)

    


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_modal_from_db_route_has_required_fields(summarizer_agent, article_doc):
    state = {
        "messages": [HumanMessage(content=f"Can you summarize the article titled '{article_doc['title']}' ?")],
    }
    result = summarizer_agent.graph.invoke(state)

    assert "messages" in result and isinstance(result["messages"][-1], AIMessage)
    assert "modals" in result and isinstance(result["modals"], list) and result["modals"]
    payload = result["modals"][-1]
    assert payload.get("type") == "summary"
    for k in ("answer", "title", "url", "summary"):
        assert k in payload and isinstance(payload[k], str) and payload[k].strip()

    # title/url match
    assert _normalize(payload["title"]) == _normalize(article_doc["title"])
    assert payload["url"].strip() == article_doc["url"].strip()
    assert _normalize(payload["answer"]) == _normalize(result["messages"][-1].content)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_modal_with_conversation_history_flow(summarizer_agent, article_doc):
    tool_call_id = f"call_{uuid.uuid4().hex}"

    state = {
        "messages": [
            HumanMessage(content="Hi, can you send me the author name of the article you told me about before?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_articles_from_database_tool",
                    "args": {"spec": {"title": article_doc["title"]}},
                    "id": tool_call_id,
                }],
            ),
            ToolMessage(
                content=f"title: {article_doc['title']}, author: {article_doc['author'][0]}",
                tool_call_id=tool_call_id,
            ),
            AIMessage(content=f"Sure! the author is - {article_doc['author'][0]}"),
            HumanMessage(content="Can you summarize this article?"),
        ],
        "user_query": "Can you summarize this article?",
    }
    result = summarizer_agent.graph.invoke(state)

    assert "modals" in result and isinstance(result["modals"], list) and result["modals"]
    payload = result["modals"][-1]
    assert payload.get("type") == "summary"
    for k in ("answer", "title", "url", "summary"):
        assert k in payload and isinstance(payload[k], str) and payload[k].strip()

    # title/url match
    assert _normalize(payload["title"]) == _normalize(article_doc["title"])
    assert payload["url"].strip() == article_doc["url"].strip()
    assert _normalize(payload["answer"]) == _normalize(result["messages"][-1].content)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_asking_clarification(summarizer_agent, article_doc):
    state = {
        "messages": [HumanMessage(content=f"Can you summarize the article you send me before ?")],
        "current_page": article_doc,
    }
    result = summarizer_agent.graph.invoke(state)

    assert "messages" in result and isinstance(result["messages"][-1], AIMessage)

    for m in result["messages"]:
        m.pretty_print()

    assert "modals" in result and isinstance(result["modals"], list) and result["modals"]
    payload = result["modals"][-1]
    assert payload.get("type") == "summary"
    for k in ("answer", "title", "url", "summary"):
        assert k in payload 
    # We expect the chosen article; allow minor whitespace differences in title
    assert _normalize(payload["title"]) == ""
    assert payload["url"].strip() == ""
    assert _normalize(payload["answer"]) == _normalize(result["messages"][-1].content)

    
    
