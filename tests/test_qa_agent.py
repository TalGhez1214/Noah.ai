import pytest
from agents.sub_agents.qa import QASubAgent
from langchain.output_parsers import ResponseSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rag.rag_piplines.articles_finder_graph import build_graph
from agents.prompts import QA_PROMPT


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def rag_retriever():
    # Your existing graph; in tests we will monkeypatch agent.invoke, so the retriever won't actually run
    return build_graph()

@pytest.fixture
def qa_agent(rag_retriever):
    return QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=QA_PROMPT)

@pytest.fixture
def current_page_article():
    """Mock article content for the current page."""
    return {
        "content": (
            "Artificial intelligence (AI) refers to systems that perform tasks typically requiring human intelligence, "
            "such as recognizing patterns, understanding language, or making decisions. In this article, we explain "
            "how modern AI models learn from large datasets, why fine-tuning helps align behavior to specific tasks, "
            "and what limitations remain around reliability and bias."
        )
    }


# ---------------------------
# Basic init and prompt tests
# ---------------------------

def test_init_sets_attributes(qa_agent, rag_retriever):
    assert qa_agent.retriever == rag_retriever
    assert qa_agent.name == "qa_agent"
    assert isinstance(qa_agent._tools, list)
    assert hasattr(qa_agent.agent, "invoke")

def test_format_system_prompt_inserts_content_and_instructions(qa_agent):
    state = {"current_page": "PAGE CONTENT"}
    prompt = qa_agent._format_system_prompt(state)
    assert "PAGE CONTENT" in prompt
    # And the format instructions from the output parser should also be present
    assert "resource_list" in prompt and "answer" in prompt


# ---------------------------
# Structured output parsing tests
# ---------------------------

def test_structured_output_valid_json(qa_agent):
    valid_json = '{"answer": "42", "resource_list": ["1", "2"]}'
    result = qa_agent.structured_output(valid_json)
    assert result["answer"] == "42"
    assert result["resource_list"] == ["1", "2"]

def test_structured_output_coerces_non_list_resource_list(qa_agent):
    # Model mistakes like returning a string for resource_list should be coerced to []
    bad_json = '{"answer": "ok", "resource_list": "1,2"}'
    result = qa_agent.structured_output(bad_json)
    assert result["answer"] == "ok"
    assert result["resource_list"] == []


# ---------------------------
# call() behavior tests
# ---------------------------

def test_call_runs_and_returns_expected_keys(qa_agent):
    """
    Minimal sanity test: we simulate a normal agent run where the final content
    is parsed and replaced with only the answer text in the last message.
    """
    fake_msg = AIMessage(content='{"answer": "hi", "resource_list": ["1"]}')
    qa_agent.agent.invoke = lambda _: {"messages": [HumanMessage(content="Q?"), fake_msg]}

    state = {"messages": [], "current_page": {"content": "abc"}, "user_query": "Q?"}
    result = qa_agent.call(state)

    assert "messages" in result
    assert result["agent"] == "qa_agent"
    # With your updated call(), last assistant content is ONLY the answer text
    assert result["messages"][-1].content == "hi"
    # And the IDs are returned separately
    assert result["relevant_articles_for_user"] == ["1"]


def test_answer_from_current_page_no_tools(qa_agent, current_page_article, monkeypatch):
    """
    User asks about the current page's content; agent should NOT need to call tools.
    We simulate no tool messages in the trace and ensure resource_list == [].
    """
    user_q = "According to this article, what is AI?"
    final_json = {
        "answer": "AI refers to systems that perform tasks needing human-like intelligence, "
                  "such as pattern recognition and language understanding.",
        "resource_list": []
    }

    # No tool calls in the trace; just a direct answer
    def fake_invoke(_):
        return {
            "messages": [
                HumanMessage(content=user_q),
                AIMessage(content=str(final_json).replace("'", '"'))
            ]
        }

    monkeypatch.setattr(qa_agent.agent, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page_content": current_page_article["content"],
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # Verify final assistant message is only the answer text
    assert result["messages"][-1].content.startswith("AI refers to systems that perform tasks")
    # No DB resources used
    assert result["relevant_articles_for_user"] == []
    # And there were no tool messages in the trace we returned
    # (our fake_invoke didn't include any ToolMessage objects)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs == []


def test_calls_db_tool_and_returns_resource_ids(qa_agent, current_page_article, monkeypatch):
    """
    Simulate a run where the agent calls the DB tool, uses two chunk IDs,
    and returns a valid resource_list. We also drop a ToolMessage into the trace
    to assert that a tool call "happened".
    """
    user_q = "Can you cite where the article discusses model limitations?"
    used_ids = ["chunk-123", "chunk-456"]

    # Simulate a tool call followed by final answer
    def fake_invoke(_):
        return {
            "messages": [
                HumanMessage(content=user_q),
                # a pretend tool call result (what the agent would have seen from the tool)
                ToolMessage(
                    tool_call_id="call-1",
                    name="get_data_for_answer_from_database_tool",
                    content=str([
                        {"id": used_ids[0], "content": "Limitations include reliability and bias."},
                        {"id": used_ids[1], "content": "Bias can arise from data and objectives."},
                    ])
                ),
                # final agent JSON
                AIMessage(content=(
                    '{"answer": "The article notes reliability and bias as key limitations.", '
                    f'"resource_list": ["{used_ids[0]}", "{used_ids[1]}"]}}'
                ).replace("}}", "}"))
            ]
        }

    monkeypatch.setattr(qa_agent.agent, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page_content": current_page_article["content"],
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # Answer text only in the last assistant message
    assert "key limitations" in result["messages"][-1].content
    # The returned IDs are exactly the ones used
    assert result["relevant_articles_for_user"] == used_ids

    # Verify a DB tool call was present in the message trace
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "get_data_for_answer_from_database_tool" for tm in tool_msgs)


def test_off_topic_basketball_uses_web_search_and_empty_resources(qa_agent, monkeypatch):
    """
    User asks about basketball (off-website topic). The QA agent should use the web search tool.
    resource_list should be [] (no DB chunks used).
    """
    user_q = "Who won the NBA championship last season?"

    # Simulate a web_search tool call followed by final JSON that starts with
    # the required disclaimer (per your prompt) and an empty resource_list.
    def fake_invoke(_):
        return {
            "messages": [
                HumanMessage(content=user_q),
                ToolMessage(
                    tool_call_id="call-2",
                    name="web_search",
                    content=str([{"title": "NBA Finals", "url": "https://example.com", "snippet": "..." }])
                ),
                AIMessage(content=(
                    '{"answer": "I didn’t find enough relevant information on the website, so here’s what I found: '
                    'The NBA champion was ...", "resource_list": []}'
                ))
            ]
        }

    monkeypatch.setattr(qa_agent.agent, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content=user_q)],
        "current_page_content": "",  # nothing relevant on-page
        "user_query": user_q,
    }
    result = qa_agent.call(state)

    # The final message should contain the disclaimer and some answer
    assert result["messages"][-1].content.startswith("I didn’t find enough relevant information on the website")
    # No DB chunks used for off-topic question
    assert result["relevant_articles_for_user"] == []

    # Ensure web_search tool was used
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "web_search" for tm in tool_msgs)
