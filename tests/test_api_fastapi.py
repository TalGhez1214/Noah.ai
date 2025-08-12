from fastapi.testclient import TestClient
import importlib

def test_ask_endpoint(monkeypatch, stub_router_to):
    # Import app AFTER we set up monkeypatches if needed
    # but here we only need to control the manager's router inside the app.
    from main import app, manager_agent
    client = TestClient(app)

    # Force router to 'qa' and stub ManagerAgent.route() to a controlled output
    def fake_route(q):
        return "🔁 Routed to `qa` agent:\n\nANSWER"
    monkeypatch.setattr(type(manager_agent), "route", lambda self, q: fake_route(q), raising=False)

    resp = client.post("/ask", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == "ANSWER"
    assert data["agent_used"] == "qa"
