import pytest
from agents.sub_agents.qa import QASubAgent
from langchain_openai import ChatOpenAI
from agents.prompts import QA_PROMPT
from rag.rag_piplines.rag_retriever import RAGRetriever
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def rag_retriever():
    """
    Fixture to create a RAGRetriever instance.
    """
    base_path = "./rag/data_indexing/indexes_and_metadata_files"
    return RAGRetriever(base_path)


@pytest.fixture
def user_query():
    """
    Fixture to provide a sample user query.
    """
    return "Who is Benjamin Netanyahu?"


# Test the initialization of the QASubAgent
def test_qa_agent_init(rag_retriever):
    agent = QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=QA_PROMPT)
    
    assert agent.retriever == rag_retriever
    assert agent.name == "qa_agent"
    assert callable(agent.get_knowledge_for_answer)
    assert hasattr(agent.agent, "invoke")


# Test get_knowledge_for_answer tool directly
def test_qa_agent_knowledge_retrieval(rag_retriever, user_query):
    agent = QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=QA_PROMPT)
    knowledge = agent.get_knowledge_for_answer(user_query)
    
    assert isinstance(knowledge, str)
    assert len(knowledge.strip()) > 0  # Should return non-empty string

    print(f"Retrieved knowledge: {knowledge[:100]}...")  # Print first 100 chars for debug


# Test full agent behavior via .call()
def test_qa_agent_call(rag_retriever, user_query):
    agent = QASubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=QA_PROMPT)

    state = {
        "messages": [
            HumanMessage(content=user_query)
        ]
    }

    result = agent.call(state)

    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)
    
    # Print for debug
    result["messages"][-1].pretty_print()
