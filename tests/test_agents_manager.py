import pytest
from agents.manager_agent.manager_agent import ManagerAgent
from langchain.chains import LLMChain


def test_manager_routes_to_summary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LLMChain, "run", lambda self, **kwargs: "summary")
    m = ManagerAgent()
    res = m.route("summarize this topic please")
    assert "Routed to `summary`" in res


def test_manager_dynamic_agent_registration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LLMChain, "run", lambda self, **kwargs: "custom")
    m = ManagerAgent()

    class CustomAgent:
        def run(self, q: str) -> str:
            return f"custom:{q}"
        def describe(self) -> str:
            return "A custom demo agent."

    m.sub_agents["custom"] = CustomAgent()
    res = m.route("hello")
    assert "Routed to `custom`" in res
    assert "custom:hello" in res



def test_manager_propagates_subagent_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LLMChain, "run", lambda self, **kwargs: "qa")
    m = ManagerAgent()

    def _boom(_q):
        raise RuntimeError("subagent failure")
    m.sub_agents["qa"].run = _boom

    with pytest.raises(RuntimeError, match="subagent failure"):
        m.route("trigger error")
