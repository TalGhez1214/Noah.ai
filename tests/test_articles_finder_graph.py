# tests/test_search_graph_end_to_end_llm.py
import os
from datetime import datetime, timedelta, timezone, date

import pytest
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage

# ---- import your graph module ----
import rag.rag_piplines.articles_finder_graph as M

# -----------------------------------------------------------------------------
# Global preconditions / markers
# -----------------------------------------------------------------------------
pytestmark = [
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                       reason="Requires OPENAI_API_KEY for analyze+embeddings"),
]


# -----------------------------------------------------------------------------
# Mongo helpers
# -----------------------------------------------------------------------------
def _mongo():
    uri = getattr(M, "MONGO_URI", os.getenv("MONGODB_URI") or os.getenv("MONGO_URI"))
    if not uri:
        pytest.skip("MONGO_URI/MONGODB_URI not configured")
    return MongoClient(uri, tz_aware=True, tzinfo=timezone.utc)

def _db(client):
    return client[getattr(M, "DB_NAME", "news")]

def _chunks_col(client):
    return _db(client)[getattr(M, "CHUNKS_COL", "test_articles_chunks")]

def _articles_col(client):
    return _db(client)[getattr(M, "ARTICLES_COL", "test_articles")]

def _skip_if_no_chunks(client):
    if _chunks_col(client).estimated_document_count() == 0:
        pytest.skip("No CHUNKS data; cannot run hybrid retrieval")

def _find_article_by_exact_title(client, title: str):
    return _articles_col(client).find_one({"title": title}, {"_id": 1, "title": 1})

def _has_author_in_chunks(client, author: str) -> bool:
    col = _chunks_col(client)
    return bool(col.find_one({
        "$or": [
            {"author": author},
            {"author": [author]},
            {"author": {"$in": [author]}},
        ]
    }, {"_id": 1}))

def _assert_results_exist_in_articles(client, results_list):
    """Ensure each returned result refers to a document in ARTICLES_COL by _id."""
    articles = _articles_col(client)
    for r in results_list:
        aid = r.get("id")
        assert aid is not None, "Result missing 'id' field"
        doc = articles.find_one({"_id": aid}, {"_id": 1})
        assert doc is not None, f"Top result {aid} not found in ARTICLES_COL"


# -----------------------------------------------------------------------------
# Make sure CE is disabled for this file (you asked to exclude CE tests)
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _disable_ce(monkeypatch):
    monkeypatch.setattr(M, "USE_CROSS_ENCODER", False, raising=False)
    monkeypatch.setattr(M, "_should_use_ce", lambda state: False, raising=True)


# -----------------------------------------------------------------------------
# Graph running helper (returns the dict that LangGraph returns)
# -----------------------------------------------------------------------------
def _run_graph(user_query: str, messages=None, file_type="article"):
    messages = messages or []
    graph = M.build_graph()
    init = M.SearchState(messages=messages, user_query=user_query, file_type=file_type)
    out = graph.invoke(init)  # <-- returns a dict in your setup
    assert isinstance(out, dict), f"Expected dict state, got {type(out)}"
    return out


# =============================================================================
# TESTS (assertions use dict-style access)
# =============================================================================

# 1) user asks articles about a topic (no messages)
def test_llm_graph_basic_topic():
    client = _mongo()
    _skip_if_no_chunks(client)

    user_query = "I want articles about AI."
    result = _run_graph(user_query)

    print(f"user_query: {user_query}")
    assert isinstance(result.get("semantic_query", ""), str) and len(result.get("semantic_query", "")) > 0
    assert isinstance(result.get("top_results", []), list) and len(result["top_results"]) > 0
    _assert_results_exist_in_articles(client, result["top_results"])

    for r in result["top_results"][:3]:  # print top 3
        print(f"- {r.get('title')} (id={r.get('id')})")


# 2) user asks 2 articles about a topic
def test_llm_graph_requested_two():
    client = _mongo()
    _skip_if_no_chunks(client)

    user_query = "Give me 2 articles about AI."
    result = _run_graph(user_query)

    assert result.get("requested_k") == 2
    assert 1 <= len(result.get("top_results", [])) <= 2
    if result.get("top_results"):
        _assert_results_exist_in_articles(client, result["top_results"])

    for r in result["top_results"][:3]:  # print top 3
        print(f"- {r.get('title')} (id={r.get('id')})")


# 3) user asks articles from a specific author in the query
def test_llm_graph_author_in_query():
    client = _mongo()
    author = "Noam Harari"
    if not _has_author_in_chunks(client, author):
        pytest.skip(f"Author '{author}' not present in CHUNKS; skipping.")

    user_query = f"Show me articles by {author} about AI."
    result = _run_graph(user_query)

    f = result.get("filters", {})
    print(f"Extracted filters: {f}")
    assert f.get("author"), "Expected author filter to be extracted"
    assert author in f["author"]

    assert len(result.get("top_results", [])) > 0
    _assert_results_exist_in_articles(client, result["top_results"])
    for r in result["top_results"]:
        ra = r.get("author")
        if ra:
            assert ra == [author]

    for r in result["top_results"][:3]:  # print top 3
        print(f"- {r.get('title')} by {r.get('author')}")

    


# 4) user asks 1 article for the author referenced in previous messages
def test_llm_graph_author_from_context_one():
    client = _mongo()
    author = "Lior Ben-David"
    if not _has_author_in_chunks(client, author):
        pytest.skip(f"Author '{author}' not present in CHUNKS; skipping.")

    messages = [
        HumanMessage(content=f"Earlier we looked at {author}'s work."),
        AIMessage(content="Yes, we discussed some of his pieces."),
    ]
    user_query = "Show me one article by him about AI."
    result = _run_graph(user_query, messages=messages)

    assert result.get("requested_k") == 1
    f = result.get("filters", {})
    print(f"Extracted filters: {f}")
    assert f.get("author"), "Expected author to be picked from context"
    assert isinstance(f["author"], list) and len(f["author"]) == 1
    assert author in f["author"]
    assert 1 <= len(result.get("top_results", [])) <= 1
    if result.get("top_results"):
        _assert_results_exist_in_articles(client, result["top_results"])
        assert result["top_results"][0].get("author") == [author]
    
    for r in result["top_results"][:3]:  # print top 3
        print(f"- {r.get('title')} by {r.get('author')}")


# 5) user asks a specific article by exact title
def test_llm_graph_exact_title_bias():
    client = _mongo()
    title = "Open community proposes SPDX-style manifests for AI datasets"
    if not _find_article_by_exact_title(client, title):
        pytest.skip(f"Title not found: '{title}'")

    user_query = f'I’m looking for the article titled "{title}".'
    result = _run_graph(user_query)

    assert result.get("requested_k") == 1
    assert len(result.get("top_results", [])) == 1
    _assert_results_exist_in_articles(client, result["top_results"])
    assert (result["top_results"][0].get("title") == title)


# 6) user asks for articles from the last 2 months (explicit ISO dates)
def test_llm_graph_last_two_months():
    """
    User asks with a relative timeframe and no explicit dates.
    We assert the analyzer resolves a 'from' date close to today-~60 days,
    and that results exist and map back to real ARTICLES docs.
    """
    client = _mongo()
    _skip_if_no_chunks(client)  # make sure CHUNKS has data

    user_query = "Please give me AI articles from the last 2 months."
    result = _run_graph(user_query)  # same helper you used in other tests

    # --- analyzer expectations ---
    # Model should produce a semantic query and filters with a 'from' date.
    assert isinstance(result.get("semantic_query", ""), str) and len(result["semantic_query"]) > 0

    filters = result.get("filters", {})
    frm = filters.get("from")
    to  = filters.get("to")  # may be None or today's date; both ok

    assert frm is not None, "Expected 'from' to be populated for 'last 2 months'"

    # Convert to date and check it's ~60 days ago (allow small slack for month length)
    today = datetime.now(timezone.utc).date()
    from_dt = datetime.fromisoformat(frm).date()

    # Accept between 50 and 70 days ago to be robust to model interpretation of "months"
    lower_bound = today - timedelta(days=70)
    upper_bound = today - timedelta(days=50)
    assert lower_bound <= from_dt <= upper_bound, f"'from'={from_dt} not within expected 50–70 days window"

    if to is not None:
        to_dt = datetime.fromisoformat(to).date()
        # to should be "near today"
        assert to_dt <= today and (today - to_dt).days <= 3, f"'to'={to_dt} should be today or very recent"

    # --- results expectations ---
    tops = result.get("top_results", [])
    assert isinstance(tops, list) and len(tops) > 0, "Expected at least one result"

    # Ensure each returned item corresponds to a real ARTICLES doc
    _assert_results_exist_in_articles(client, tops)

    # Optional: sanity-check recency bias actually prioritized newer results
    # (non-strict, just ensure most top docs are within ~70 days)
    recentish = 0
    for r in tops[:3]:
        pub = r.get("published_at")
        if isinstance(pub, str):
            try:
                pub = datetime.fromisoformat(pub)
            except Exception:
                pub = None
        if isinstance(pub, datetime):
            if (today - pub.date()).days <= 70:
                recentish += 1
    assert recentish >= 1, "Expected at least one fairly recent article in the top results"

    # Print top 3
    for r in tops[:3]:
        print(f"- {r.get('title')} by {r.get('author')}, published at {r.get('published_at')}")


# 7) user asks for articles from an author that does not exist
def test_llm_graph_unknown_author_behavior():
    client = _mongo()
    _skip_if_no_chunks(client)

    user_query = "Show me AI articles by richard gear."
    result = _run_graph(user_query)
    assert result.get("filters", {}).get("author") == []
    assert any(
        "Author filter removed (no matches on chunks)." in w for w in (result.get("filter_warnings") or [])
    )
    if result.get("top_results"):
        _assert_results_exist_in_articles(client, result["top_results"])

    print(f"{result.get('semantic_query')}")
    print(f"{result.get('filters')}")
    

# 8) user asks for articles before two years ago → expect none (if dataset recent)
def test_llm_graph_before_two_years_none():
    client = _mongo()
    _skip_if_no_chunks(client)

    # Compute the “~2 years ago” cutoff using days (robust across leap years).
    today = datetime.now(timezone.utc).date()
    approx_two_years_ago = today - timedelta(days=730)

    # If your dataset actually has older docs than ~2 years, skip this test
    # because this test asserts that the graph returns no results.
    older_exists = _chunks_col(client).find_one({
        "published_at": {
            "$lt": datetime(
                approx_two_years_ago.year,
                approx_two_years_ago.month,
                approx_two_years_ago.day,
                tzinfo=timezone.utc
            )
        }
    })
    if older_exists:
        pytest.skip("Older docs exist; this case assumes none exist.")

    user_query = "Give me AI articles that was published before 2 years or more."
    result = _run_graph(user_query)
    print(f"Filters extracted: {result.get('filters')}")

    # ---- analyzer expectations ----
    f = result.get("filters", {})
    # For “before 2 years”, we expect a 'to' date near ~2 years ago, and no 'from'
    assert f.get("to") is not None, "Expected 'to' for a 'before X years' query"
    assert f.get("from") in (None, ), "Expected 'from' to be null for an open-ended 'before' request"

    # Tolerant check: allow +- ~10 days around 730 for LLM interpretation / calendar month variability
    to_dt = datetime.fromisoformat(f["to"]).date()
    delta_days = (today - to_dt).days
    assert 720 <= delta_days <= 740, f"Expected 'to' ~2 years ago; got {to_dt} (Δ={delta_days} days)"

    # ---- results expectations ----
    # Since we asserted there are no docs older than that, we expect no results.
    assert result.get("top_results", []) == []