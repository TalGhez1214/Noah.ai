# tests/test_summarizer_agent.py
import os
import re
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.sub_agents.summarizer.summarizer_agent import SummarizerSubAgent
from agents.prompts import SUMMARY_PROMPT
from dotenv import load_dotenv
from pymongo import MongoClient
import rag.rag_piplines.articles_finder_graph as M
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")
# ---------- helpers ----------
def _tool_map(agent):
    # Use the exposed list from the sub-agent, not the compiled graph
    return {t.name: t for t in getattr(agent, "tools", [])}


@pytest.fixture
def mongo_collection():
    if not MONGO_URI:
        return None
    client = MongoClient(MONGO_URI)
    db = client[DB]
    collection = db[COLL]
    yield collection
    client.close()

@pytest.fixture
def summarizer_agent(mongo_collection):
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
        "content": article_content,  # trimmed for brevity
        "fetched_at": "2025-08-08T09:39:23.425560+00:00",
        "published_at": "2025-08-09T16:45:00Z",
        "section": None,            # <-- was `null`
        "source": "www.timesofisrael.com",
        "title": "Ex-   PM aide crafted pro-Qatar messages, sent to Urich, Feldstein to publish in media",
    }

# ---------- tests ----------
def test_summarizer_agent_init(summarizer_agent):
    assert summarizer_agent.name == "summary_agent"
    assert hasattr(summarizer_agent.agent, "invoke")
    tm = _tool_map(summarizer_agent)
    assert "summary_content_from_link_tool" in tm
    assert "summary_article_from_current_page_tool" in tm
    assert "get_articles_from_database_tool" in tm or "get_articles_for_summary_from_database_tool" in tm

def test_link_tool_returns_content_from_mongo(summarizer_agent, article_doc):
    tm = _tool_map(summarizer_agent)
    tool = tm["summary_content_from_link_tool"]
    out = tool.run(f"please summarize {article_doc['url']}")
    assert isinstance(out, dict)
    assert article_doc["title"] == out["title"]

def test_current_page_tool_reads_injected_state(summarizer_agent, article_doc):
    tm = _tool_map(summarizer_agent)
    tool = tm["summary_article_from_current_page_tool"]
    state = {"current_page": article_doc}
    # Call the underlying function to bypass InjectedState (unit test context)
    out = tool.func(state)
    assert isinstance(out, dict)
    assert "Yisrael Einhorn, a former campaign adviser" in out["content"]

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_summarizer_agent_call_link_route_integration(summarizer_agent, article_doc):
    state = {"messages": [HumanMessage(content=f"Can you summarize {article_doc['url']} ?")]}
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)

    for m in result["messages"]:
        m.pretty_print()

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_summarizer_agent_call_current_page_route_integration(summarizer_agent, article_doc):
    state = {
        "messages": [HumanMessage(content="Can you send me a summary of this article?.")],
        "current_page": article_doc,
    }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)


    for m in result["messages"]:
        m.pretty_print()

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_summarizer_agent_call_db_route_integration_with_title(summarizer_agent, article_doc):
    state = {
        "messages": [HumanMessage(content=f"Can you summarize the article titled '{article_doc['title']}' ?")],
    }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)

    obj_hits = sum((m.content if isinstance(m.content, str) else str(m.content)).count("final_score")for m in result["messages"] if isinstance(m, ToolMessage))
    assert obj_hits == 1, f"Expected exactly 1 ObjectId in tool content, got {obj_hits}"

    for m in result["messages"]:
        m.pretty_print()
    

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_summarizer_agent_call_db_route_integration_with_description(summarizer_agent, article_doc):
    short_desc = "the former campaign adviser to ex-prime minister Benjamin Netanyahu, crafted pro-Qatar messages and sent them to journalists Urich and Feldstein for publication."
    state = {
        "messages": [HumanMessage(content=f"Can you summarize the article I saw yesterday about {short_desc}?")],
    }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)
    assert article_doc["title"] in result["messages"][-2].content

    obj_hits = sum((m.content if isinstance(m.content, str) else str(m.content)).count("final_score")for m in result["messages"] if isinstance(m, ToolMessage))
    assert obj_hits == 1, f"Expected exactly 1 ObjectId in tool content, got {obj_hits}"

    for m in result["messages"]:
        m.pretty_print()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_summarizer_agent_call_db_route_integration_with_conversation_history(summarizer_agent, article_doc):
    """
    This test simulate the case where the user is asking about an article mentioned before in the conversation.
    """
    import json, uuid
    tool_call_id = f"call_{uuid.uuid4().hex}"

    state = {
            "messages": [
                HumanMessage(content="Hi, can you send me the author name of the article you told me about before?"),

                # 1) The AI asks to call a tool (this is what the agent would normally produce)
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_articles_from_database_tool",
                        "args": {"spec": {"title": article_doc["title"]}},  # Wrapped in 'spec' as required
                        "id": tool_call_id,
                    }],
                ),

                # 2) The tool's response (what you're trying to craft)
                ToolMessage(
                    content=f"title: {article_doc['title']}, author: {article_doc['author'][0]}",  # ALWAYS a string
                    tool_call_id=tool_call_id,
                ),

                # 3) The AI responds using the tool output
                AIMessage(content=f"Sure! the author is - {article_doc['author'][0]}"),

                HumanMessage(content="Can you summarize this article?"),
            ],
            "user_query": "Can you summarize this article?",
        }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)
    assert article_doc["title"] in result["messages"][-2].content

    obj_hits = sum((m.content if isinstance(m.content, str) else str(m.content)).count("final_score")for m in result["messages"] if isinstance(m, ToolMessage))
    assert obj_hits == 1, f"Expected exactly 1 ObjectId in tool content, got {obj_hits}"

    for m in result["messages"]:
        m.pretty_print()


def test_summerizer_agent_extract_artical_from_previous_message(summarizer_agent, article_doc):
    """
    Test the ability of the summarizer agent to extract article content from previous messages in the conversation
    without calling any tools.
    """
    import json, uuid
    tool_call_id = f"call_{uuid.uuid4().hex}"
    
    state = {
        "messages": [

            HumanMessage(content="Hi, can you send me the title of the article about Yisrael Einhorn?"),
            AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_articles_from_database_tool",
                        "args": {"spec": {"article_content": "Yisrael Einhorn, a former campaign adviser"}},  # Corrected field name
                        "id": tool_call_id,
                    }],
                ),
            ToolMessage(
                content= json.dumps(article_doc),
                    tool_call_id=tool_call_id,
                ),
            AIMessage(content=f"Sure! the title is - {article_doc['title']}"),

            HumanMessage(content="Can you summarize this article?"),
        ]
    }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)
    assert len([msg for msg in result["messages"] if isinstance(msg, ToolMessage)]) == 1 # Make sure no additional tool calls were made

    for m in result["messages"]:
        m.pretty_print()



def test_summerizer_agent_clarification_question(summarizer_agent):
    """
    Test the ability of the summarizer agent to ask clarifying questions
    """
    state = {
        "messages": [
            HumanMessage(content="Can you summarize for me the article about Bibi Gantz?"),
        ]
    }
    result = summarizer_agent.call(state)
    assert "messages" in result
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str)

    for m in result["messages"]:
        m.pretty_print()



