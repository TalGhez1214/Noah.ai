import pytest
from agents.sub_agents.articles_finder_agent import ArticalFinderAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agents.prompts import ARTICLES_FINDER_PROMPT
from rag.rag_piplines.rag_retriever import RAGRetriever
from langchain_core.messages import HumanMessage, SystemMessage

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
    agent = ArticalFinderAgent(rag_retriever)
    assert agent.retriever == rag_retriever
    assert isinstance(agent._llm, ChatOpenAI)
    assert agent._response_schemas == [
        ResponseSchema(name="Summary", description="One-sentence summary of the text"),
        ResponseSchema(name="Key Quote", description="A key direct quote from the text"),
    ]

# Test the build_articles_finder_agent method
def test_build_articles_finder_agent(rag_retriever, user_query):
    agent = ArticalFinderAgent(rag_retriever)
    article = {"title": "Test article","author": "Test author" ,"content": "Test content"}
    built_agent = agent.build_articles_finder_agent(user_query, article)
    assert built_agent.name == "articles_finder"


# Test the structured_output method
def test_structured_output(rag_retriever):
    agent = ArticalFinderAgent(rag_retriever)
    llm_output = "This is a summary. This is a key quote."
    structured_output = agent.structured_output(llm_output)
    assert structured_output == {"Summary": "This is a summary.", "Key Quote": "This is a key quote."}

# Test the error handling in the structured_output method
def test_structured_output_error_handling(rag_retriever):
    agent = ArticalFinderAgent(rag_retriever)
    llm_output = "Invalid output"
    output = agent.structured_output(llm_output)
    assert output["Summary"] == "Invalid output"
    assert output["Key Quote"] == "Invalid output"


# Test the get_knowledge_for_answer method
def test_get_knowledge_for_answer(rag_retriever, user_query):
    agent = ArticalFinderAgent(rag_retriever)
    knowledge = agent.get_knowledge_for_answer(user_query)
    assert len(knowledge) == 3

def test_agent_answers(rag_retriever, user_query):
    agent = ArticalFinderAgent(rag_retriever)
    articles = agent.get_knowledge_for_answer(user_query)

    for article in articles:
        agent_answer = agent.build_articles_finder_agent(user_query=user_query, article=article).invoke({"messages": [HumanMessage(content=user_query)]})
        agent_answer["messages"][-1].pretty_print()
        json_output = agent.structured_output(agent_answer["messages"][-1].content)
        assert "Summary" in json_output
        assert "Key Quote" in json_output

