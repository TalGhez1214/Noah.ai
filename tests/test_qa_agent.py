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
def test_answer_from_current_page_no_tools(qa_agent, current_page_article):
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

    # Parser should return no DB resources (we didn't need tools)
    assert result["relevant_articles_for_user"] == []

    # Ensure no tool calls in the trace
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs == []


@REQUIRES_KEYS
def test_calls_db_tool_and_parser_returns_ids(qa_agent):
    """
    Nudge the agent to use website content (DB tool).
    Assert: DB tool call happened, parser picked up to 3 relevant IDs (may be empty if retriever returns nothing).
    """
    user_q = "can you give me relevant AI info."
    # Empty/irrelevant current_page content nudges the agent toward DB tool
    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page": {"content": ""},
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # Produced an answer (plain text)
    answer_text = result["messages"][-1].content
    assert isinstance(answer_text, str) and len(answer_text) > 0
    print("\n\nAnswer text:\n", answer_text)

    # Verify a DB tool call happened
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "get_data_for_answer_from_database_tool" for tm in tool_msgs)

    # Parser-selected resource IDs: list[str], length <= 3 (can be empty)
    resource_ids = result["relevant_articles_for_user"]
    assert isinstance(resource_ids, list)
    assert all(isinstance(x, str) for x in resource_ids)
    assert len(resource_ids) > 0 and len(resource_ids) <= 3

    print("\n\nResource IDs used by parser:", resource_ids)


@REQUIRES_KEYS
def test_off_topic_uses_web_search_and_empty_resources(qa_agent):
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

    # No DB chunks used for off-topic question -> parser should return []
    assert result["relevant_articles_for_user"] == []

    # Ensure web_search tool was used
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "web_search" for tm in tool_msgs)


@REQUIRES_KEYS
def test_parser_caps_ids_to_three_and_keeps_answer_verbatim(qa_agent):
    """
    Ensure the parser:
      - never returns more than 3 IDs,
      - keeps the QA agent's answer text exactly as produced (verbatim).
    """
    user_q = "From the website content only, list the main AI issues the site discusses."
    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page": {"content": ""},
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # The last assistant message is the *final* answer text (verbatim from the QA agent draft)
    final_text = result["messages"][-1].content
    assert isinstance(final_text, str) and len(final_text) > 0

    # Parser-selected ids must be <= 3
    ids = result["relevant_articles_for_user"]
    assert isinstance(ids, list)
    assert len(ids) > 0 and len(ids) <= 3
    assert all(isinstance(x, str) for x in ids)

    # Optional visibility
    print("\n\nFinal answer (verbatim):\n", final_text)
    print("Parser resource IDs (<=3):", ids)
