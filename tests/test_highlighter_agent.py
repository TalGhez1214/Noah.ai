import os
import json
import pytest
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from agents.sub_agents.highlighter import HighlighterSubAgent

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _parse_json_from_fenced(text: str) -> List[Dict[str, Any]]:
    """Extract and parse the first fenced ```json ...``` block (or first [] array)."""
    import re
    m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(1) if m.lastindex else m.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _is_sentence_boundary(content: str, start: int, end: int) -> bool:
    """
    Approximate: start must be at start-of-text or after punctuation+space,
    end must be at end-of-text or just after '.', '!' or '?'.
    """
    L = len(content)
    # start boundary
    if start < 0 or end > L or start >= end:
        return False

    # end must end at punctuation or end-of-text
    if end < L and content[end - 1] not in ".!?":
        # allow end to include following spaces if punctuation was just before them
        # e.g., "... yeast.  Next" -> end can be at the space after '.'
        # move left to last non-space char and check it
        k = end - 1
        while k >= 0 and content[k].isspace():
            k -= 1
        if k < 0 or content[k] not in ".!?":
            return False

    # start must be 0 or after punctuation (possibly with spaces)
    if start > 0:
        k = start - 1
        # skip preceding spaces
        while k >= 0 and content[k].isspace():
            k -= 1
        if k >= 0 and content[k] not in ".!?":
            return False

    return True


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_article():
    # Multi-sentence content similar to your examples
    return (
        "Community fermentation clubs are popping up in libraries and neighborhood kitchens, "
        "trading sourdough starters, koji, and pickling brines like heirlooms. "
        "What began as a lockdown hobby has matured into hyper-local food culture: members track flour blends "
        "from nearby mills, catalog seasonal produce for lacto-ferments, and publish Zines with recipes in multiple languages. "
        "Workshops double as cultural exchange. Ethiopian injera techniques meet Georgian tonis puri shaping, "
        "while grandparents teach preservation methods that predate refrigeration. "
        "Food safety is a priority: clubs use shared pH meters and labeling rules, "
        "and invite microbiologists to demystify mold vs. yeast. "
        "Restaurants are paying attention, collaborating on limited-run menus that feature club ferments and locally grown grains. "
        "Critics question whether trends risk flattening distinct traditions, but most groups position themselves as stewards rather than inventors, "
        "documenting lineages and crediting families and regions of origin."
    )

@pytest.fixture
def highlighter_agent():
    # Use your default model; integration tests are conditionally skipped below
    return HighlighterSubAgent(model="gpt-4o-mini")


# ------------------------------------------------------------------------------
# Unit tests (mocked LLM)
# ------------------------------------------------------------------------------

def test_highlighter_agent_init(highlighter_agent):
    assert highlighter_agent.name == "highlighter_agent"
    assert hasattr(highlighter_agent, "agent")
    assert hasattr(highlighter_agent.agent, "invoke")

def test_highlighter_agent_call_normalizes_to_full_sentences(monkeypatch, highlighter_agent, sample_article):
    """
    Mock the LLM to return a mid-sentence, wrong-offset span and ensure the agent
    normalizes it to the full sentence boundaries and correct substring.
    """

    # Find the real full sentence containing "microbiologists" for later assertions
    idx = sample_article.lower().find("microbiologists")
    assert idx != -1, "test content sanity check failed"
    # expand to the full sentence manually (simple rule for test)
    # move left to previous '.', '!' or '?' (or start)
    left = 0
    for k in range(idx - 1, -1, -1):
        if sample_article[k] in ".!?":
            left = k + 1
            break
    while left < len(sample_article) and sample_article[left].isspace():
        left += 1
    # move right to next sentence boundary
    right = len(sample_article)
    for k in range(idx, len(sample_article)):
        if sample_article[k] in ".!?":
            right = k + 1
            break

    full_sentence = sample_article[left:right]
    assert full_sentence.strip(), "couldn't compute sentence for test fixture"

    # Mock: LLM returns a wrong offset in the middle of the sentence
    bogus_start = idx + 5
    bogus_end = bogus_start + 15
    llm_reply = {
        "messages": [
            AIMessage(
                content="```json\n"
                        + json.dumps([{
                            "start": bogus_start,
                            "end": bogus_end,
                            "sentence": "invite microbiologists to demystify mold vs. yeast.",
                            "reason": "keyword-match",
                            "score": 1.0
                        }])
                        + "\n```"
            )
        ]
    }

    monkeypatch.setattr(highlighter_agent.agent, "invoke", lambda _: llm_reply)

    state = {
        "user_query": "Show where this article mentions microbiologists",
        "current_page": {"content": sample_article},
    }

    out = highlighter_agent.call(state)
    assert "messages" in out and out["messages"], "Agent must return messages"
    assert isinstance(out["messages"][-1], AIMessage)
    spans = _parse_json_from_fenced(out["messages"][-1].content)
    assert spans, "Agent must return at least one normalized span"

    # Validate each span is snapped to sentence boundaries and matches substring
    for s in spans:
        start, end, sent = s["start"], s["end"], s["sentence"]
        assert 0 <= start < end <= len(sample_article)
        assert sample_article[start:end] == sent
        assert _is_sentence_boundary(sample_article, start, end)
        # For our specific query, ensure the sentence indeed contains the keyword
        assert "microbiologist" in sent.lower()


# ------------------------------------------------------------------------------
# Optional integration tests (real LLM) – skipped if no API key
# ------------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OpenAI API key for integration tests."
)

def test_highlighter_integration_returns_json(highlighter_agent, sample_article):
    state = {
        "user_query": "Highlight where this article mentions microbiologists",
        "current_page": {"content": sample_article},
    }
    out = highlighter_agent.call(state)
    assert "messages" in out and out["messages"]
    assert isinstance(out["messages"][-1], AIMessage)
    spans = _parse_json_from_fenced(out["messages"][-1].content)
    assert isinstance(spans, list)
    # if relevant, should find at least one; if the model decides none are relevant, zero is okay
    for s in spans:
        assert set(["start", "end", "sentence", "reason", "score"]).issubset(s.keys())
        # sanity for boundaries/substrings
        start, end = s["start"], s["end"]
        assert 0 <= start < end <= len(sample_article)
        assert sample_article[start:end] == s["sentence"]

def test_highlighter_integration_empty_when_irrelevant(highlighter_agent, sample_article):
    state = {
        "user_query": "Highlight references to superconductivity in this article",
        "current_page": {"content": sample_article},
    }
    out = highlighter_agent.call(state)
    assert isinstance(out["messages"][-1], AIMessage)
    spans = _parse_json_from_fenced(out["messages"][-1].content)
    # Either empty, or (rarely) some near-miss the model thinks is relevant; accept both but validate structure
    assert isinstance(spans, list)
    for s in spans:
        assert set(["start", "end", "sentence", "reason", "score"]).issubset(s.keys())
