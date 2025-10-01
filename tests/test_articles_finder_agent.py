import pytest
from agents.sub_agents.article_finder.articles_finder_agent import ArticalFinderSubAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agents.prompts import article_finder_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.rag_piplines.articles_finder_graph import build_graph

@pytest.fixture
def rag_retriever():
    """
    Fixture to create a RAGRetriever instance.

    Returns:
        RAGRetriever: An instance of the RAGRetriever class.
    """
    return build_graph()

@pytest.fixture
def user_query():
    """
    Fixture to provide a sample user query.

    Returns:
        str: A sample user query.
    """
    return "give me articles about situation in Gaza?"

# Test the initialization of the ArticalFinderAgent
def test_artical_finder_agent_init(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    assert agent.retriever == rag_retriever
    assert agent.name == "articles_finder_agent"


# Test the structured_output method
def test_structured_output(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    llm_output = "This is a summary - I love chocolate. This is a key quote from the article you need to quote - 'amazing chocolate'."
    chain = agent.prompt | agent.parser_model
    structured_output = chain.invoke({"messages": [AIMessage(content=llm_output)],
                                        "user_query": "",
                                        "title": "",
                                        "author": "",
                                        "content": llm_output,
                                        })
    # Convert to dict safely
    payload = structured_output.model_dump() if hasattr(structured_output, "model_dump") else dict(structured_output)
    assert "summary" in payload
    assert "quote" in payload
    assert payload["summary"] != ""
    assert payload["quote"] != ""

    print(payload)

# Test the error handling in the structured_output method
def test_structured_output_error_handling(rag_retriever):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    llm_output = "Invalid output"
    chain = agent.prompt | agent.parser_model
    structured_output = chain.invoke({"messages": [AIMessage(content=llm_output)],
                                        "user_query": "",
                                        "title": "",
                                        "author": "",
                                        "content": llm_output,
                                        })
    # Convert to dict safely
    payload = structured_output.model_dump() if hasattr(structured_output, "model_dump") else dict(structured_output)
    assert "summary" in payload
    assert "quote" in payload
    assert payload["summary"] == ""
    assert payload["quote"] == ""

    print(payload)

def test_agent_answers(rag_retriever, user_query):
    agent = ArticalFinderSubAgent(retriever=rag_retriever, model="gpt-4o-mini", prompt=article_finder_prompt)
    state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
    }
    agent_answer = agent.call(state)
    assert "messages" in agent_answer
    assert isinstance(agent_answer["messages"][-1], AIMessage)
    assert isinstance(agent_answer["messages"][-1].content, str)

    assert agent_answer["ui_items"] is not None
    assert agent_answer["ui_items"]["type"] == "articles"
    assert agent_answer["ui_items"]["data"][0]["quote"] != ""
    assert agent_answer["ui_items"]["data"][0]["summary"] != ""


    

