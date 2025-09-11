# tests/test_supervisor_agent_routing.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# We patch inside the module where ManagerAgent is defined
import agents.manager_agent.manager_agent as manager_mod


# ---------- Helpers ----------

def _tool_messages(messages):
    """Return all tool messages (dicts with role='tool' OR ToolMessage instances)."""
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            out.append(m)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "name": getattr(m, "name", None), "content": m.content})
    return out


# ---------- Tests ----------

def test_manager_initialization_smoke():
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="hello", user_id="u-init")
    assert hasattr(mgr, "app")
    msgs = mgr.chat()

    tools = _tool_messages(msgs)
    assert tools[0]["name"] == "transfer_to_fallback_agent"


def test_routes_to_summary_agent():
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="Summarize this article", user_id="u-sum")
    msgs = mgr.chat()

    tools = _tool_messages(msgs)
    assert tools[0]["name"] == "transfer_to_summary_agent"


def test_routes_to_articles_finder_agent():
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="Find good articles about AI chips", user_id="u-find")
    msgs = mgr.chat()

    tools = _tool_messages(msgs)
    assert tools[0]["name"] == "transfer_to_articles_finder_agent"


def test_routes_to_fallback_agent():
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="Help me write JavaScript code", user_id="u-fb")
    msgs = mgr.chat()

    tools = _tool_messages(msgs)
    assert tools[0]["name"] == "transfer_to_fallback_agent"


def test_followup_returns_to_same_agent_after_clarify():
    """
    Turn 1: route to articles_finder_agent which asks a clarifying question.
    Turn 2: short author reply should route BACK to articles_finder_agent.
    """
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="Find articles about compute", user_id="u-follow1")
    msgs1 = mgr.chat()

    tools1 = _tool_messages(msgs1)
    assert tools1[0]["name"] == "transfer_to_articles_finder_agent"

    # Follow-up reply (short name) → should route back to finder
    mgr.user_query = "please only of the author - Noam Harari"
    msgs2 = mgr.chat()

    tools2 = _tool_messages(msgs2)
    # Now should be two tool calls across the thread; check LAST one
    assert tools2[-1]["name"] == "transfer_to_articles_finder_agent"


def test_followup_summarize_after_results_routes_to_summary():
    """
    Turn 1: route to articles_finder_agent that returns results.
    Turn 2: 'Summarize the first one' → route to summary_agent.
    """
    from agents.manager_agent.manager_agent import ManagerAgent
    mgr = ManagerAgent(user_query="Find list of articles about GPUs", user_id="u-follow2")
    msgs1 = mgr.chat()

    tools1 = _tool_messages(msgs1)
    assert tools1[0]["name"] == "transfer_to_articles_finder_agent"

    # Follow-up: summarize based on results → summary agent
    mgr.user_query = "Summarize the first one"
    msgs2 = mgr.chat()

    tools2 = _tool_messages(msgs2)
    assert tools2[-1]["name"] == "transfer_to_summary_agent"
