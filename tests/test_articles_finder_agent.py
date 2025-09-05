import pytest
from agents.sub_agents.articles_finder import ArticalFinderSubAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agents.prompts import article_finder_prompt
from rag.rag_piplines.rag_retriever import RAGRetriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

@pytest.fixture
def rag_retriever():
    """
    Fixture to create a RAGRetriever instance.

    Returns:
        RAGRetriever: An instance of the RAGRetriever class.
    """
    base_path = "./rag/data_indexing/indexes_and_metadata_files"
    return RAGRetriever(base_path)

@pytest.fixture
def user_query():
    """
    Fixture to provide a sample user query.

    Returns:
        str: A sample user query.
    """
    return "What is the situation in Gaza?"

# Test the initialization of the ArticalFinderAgent
def test_artical_finder_agent_init(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    assert agent.retriever == rag_retriever
    assert agent._response_schemas == [
        ResponseSchema(name="Summary", description="One-sentence summary of the text"),
        ResponseSchema(name="Key Quote", description="A key direct quote from the text"),
    ]
    assert agent.name == "articles_finder_agent"


# Test the structured_output method
def test_structured_output(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    llm_output = "This is a summary - I love chocolate. This is a key quote - amazing chocolate."
    structured_output = agent.structured_output(llm_output)
    assert "Summary" in structured_output
    assert "Key Quote" in structured_output

# Test the error handling in the structured_output method
def test_structured_output_error_handling(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    llm_output = "Invalid output"
    output = agent.structured_output(llm_output)
    assert output["Summary"] == "Invalid output"
    assert output["Key Quote"] == "Invalid output"


# Test the get_knowledge_for_answer method
def test_get_knowledge_for_answer(rag_retriever, user_query):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    knowledge = agent.get_knowledge_for_answer(user_query)
    assert len(knowledge) == 3

def test_agent_answers(rag_retriever, user_query):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    state = {
        "user_query": user_query,
    }
    agent_answer = agent.call(state)
    assert "messages" in agent_answer
    assert isinstance(agent_answer["messages"][-1], AIMessage)
    assert isinstance(agent_answer["messages"][-1].content, str)

    agent_answer["messages"][-1].pretty_print()
    

