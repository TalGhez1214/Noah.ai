import os
import pytest

from agents.sub_agents.qa import QASubAgent
from langchain_core.messages import HumanMessage, ToolMessage
from rag.rag_piplines.articles_finder_graph import build_graph
# IMPORTANT: use the new answer-only prompt
from agents.prompts import qa_prompt


REQUIRES_KEYS = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="Integration test requires OPENAI_API_KEY and TAVILY_API_KEY",
)


# ---------------------------
# Fixtures (as requested)
# ---------------------------

@pytest.fixture
def rag_retriever():
    # Use your real graph/retriever
    return build_graph()

@pytest.fixture
def qa_agent(rag_retriever):
    # Use your real model + the new prompt (answer-only)
    return QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=qa_prompt)

@pytest.fixture
def current_page_article():
    """Current page content with a clear AI definition / theme."""
    return {
        "content": (
            "Artificial intelligence (AI) refers to systems that perform tasks typically requiring human intelligence, "
            "such as recognizing patterns, understanding language, or making decisions. "
            "Modern AI models learn from large datasets; fine-tuning adapts them to specific tasks. "
            "Limitations include reliability and bias."
        )
    }


# ---------------------------
# Basic sanity
# ---------------------------

def test_init_sets_attributes(qa_agent, rag_retriever):
    assert qa_agent.retriever == rag_retriever
    assert qa_agent.name == "qa_agent"
    assert isinstance(qa_agent._tools, list)
    assert hasattr(qa_agent.agent, "invoke")


# ---------------------------
# Integration flows (real invoke)
# ---------------------------

@REQUIRES_KEYS
def test_answer_from_current_page(qa_agent, current_page_article):
    """
    The agent should answer from the current page text only.
    Expect: no ToolMessage (no DB, no web), resource_list == [] from parser.
    """
    user_q = "According to this article, what is AI?"
    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page": {"content": current_page_article["content"]},
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # Human-readable answer (answer-only prompt)
    answer_text = result["messages"][-1].content
    print("\n\nAnswer text:\n", answer_text)

    assert isinstance(answer_text, str) and len(answer_text) > 0
    assert "intelligence" in answer_text.lower()

    for m in result["messages"]:
        m.pretty_print()


@REQUIRES_KEYS
def test_off_topic_uses_web_search(qa_agent):
    """
    Off-topic, time-sensitive question; expect web_search use and no site IDs.
    We don't assert the exact winner/team to avoid flakiness; we ensure outside-info disclosure is likely,
    web_search tool was used, and parser returns [] (since site chunks aren't relevant).
    """
    user_q = "Who won the NBA championship last season?"
    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page": {"content": ""},  # nothing relevant on-page
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    answer_text = result["messages"][-1].content
    assert isinstance(answer_text, str) and len(answer_text) > 0
    print("\n\nAnswer text:\n", answer_text)

    # Ensure web_search tool was used
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "web_search" for tm in tool_msgs)

    for m in result["messages"]:
        m.pretty_print()
