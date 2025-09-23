import pytest
from agents.sub_agents.qa import QASubAgent
from langchain.output_parsers import ResponseSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rag.rag_piplines.articles_finder_graph import build_graph
from agents.prompts import QA_PROMPT

@pytest.fixture
def rag_retriever():
    return build_graph()

@pytest.fixture
def qa_agent(rag_retriever):
    return QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=QA_PROMPT)

def test_init_sets_attributes(qa_agent, rag_retriever):
    assert qa_agent.retriever == rag_retriever
    assert qa_agent.name == "qa_agent"
    assert isinstance(qa_agent._tools, list)
    assert hasattr(qa_agent.agent, "invoke")

def test_format_system_prompt_inserts_content_and_instructions(qa_agent):
    state = {"current_page": "PAGE CONTENT"}
    prompt = qa_agent._format_system_prompt(state)
    assert "PAGE CONTENT" in prompt

def test_structured_output_valid_json(qa_agent):
    valid_json = '{"answer": "42", "resource_list": ["1", "2"]}'
    result = qa_agent.structured_output(valid_json)
    assert result["answer"] == "42"
    assert result["resource_list"] == ["1", "2"]

def test_call_runs_and_returns_expected_keys(qa_agent):
    # Patch agent.agent.invoke to simulate LLM output
    fake_msg = AIMessage(content='{"answer": "hi", "resource_list": ["1"]}')
    qa_agent.agent.invoke = lambda _: {"messages": [HumanMessage(content="Q?"), fake_msg]}
    state = {"messages": [], "current_page_content": "abc", "user_query": "Q?"}
    result = qa_agent.call(state)
    assert "messages" in result
    assert result["agent"] == "qa_agent"
    assert isinstance(result["messages"][-1].content, str)
    assert '"answer": "hi"' in result["messages"][-1].content