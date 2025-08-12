import types
import pytest

class FakeRetriever:
    """Mimics your RAGRetriever.retrieve() signature and return format."""
    def __init__(self):
        self.last_kwargs = None
        # Minimal tuple format: (final_score, id, sim, recency_weight, metadata_dict)
        self._hits = [
            (0.95, 10, 0.97, 0.98, {"title": "Item A", "url": "https://a", "text": "A text"}),
            (0.90, 11, 0.95, 0.95, {"title": "Item B", "url": "https://b", "text": "B text"}),
        ]

    def retrieve(self, question, mode="auto", k_initial_matches=80, k_final_matches=6):
        self.last_kwargs = {
            "question": question, "mode": mode,
            "k_initial_matches": k_initial_matches, "k_final_matches": k_final_matches
        }
        return list(self._hits)


@pytest.fixture()
def fake_retriever():
    return FakeRetriever()


@pytest.fixture()
def stub_chain_run(monkeypatch):
    """
    Monkeypatch LLMChain.run to a deterministic function that returns
    the formatted inputs so we can assert on them.
    """
    def _stub_chain_run(self, **kwargs):
        # return something short and deterministic for assertions
        q = kwargs.get("question", "")
        ctx = kwargs.get("context", "").replace("\n", " ")[:120]
        return f"[LLMStub] q={q!r} ctx~={ctx!r}"
    import langchain
    # Patch only for our tests; identify LLMChain class by import path
    from langchain.chains import LLMChain
    monkeypatch.setattr(LLMChain, "run", _stub_chain_run, raising=True)
    return True


@pytest.fixture()
def stub_router_to(monkeypatch):
    """
    Returns a helper that forces ManagerAgent.router_chain to return a given key.
    """
    def _force(manager, key: str):
        # replace router_chain.run to fixed key
        def _stub_run(*args, **kwargs):
            return key
        monkeypatch.setattr(manager.router_chain, "run", _stub_run, raising=True)
    return _force
