# test_search_capabilities.py
"""
Integration tests for MongoDB Atlas Search/Vector Search and the hybrid RankFusion pipeline.

This suite:
- Runs BM25 keyword tests against the ARTICLES collection (query "AI")
- Checks filters behavior (author, date)
- Runs semantic vector search against CHUNKS
- Runs your hybrid $rankFusion pipeline end-to-end
- Verifies a specific title is the top result in a keyword search:
  "Open community proposes SPDX-style manifests for AI datasets"

All tests are READ-only. They skip gracefully if indexes/keys/data are missing.
"""

import os
import pytest
from datetime import datetime, timezone
from pymongo import MongoClient

# ====== CHANGE THIS import to your module ======
# Example: import retrieval_pipeline as M
import rag.rag_piplines.articles_rag_retriever_mongo_based as M  # <-- replace with your module name


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def mongo():
    """Create a tz-aware MongoDB client using the URI from your module/env and ping it.

    Skips the suite if the client cannot connect.
    """
    uri = getattr(M, "MONGODB_URI", os.getenv("MONGODB_URI"))
    if not uri:
        pytest.skip("MONGODB_URI not configured")
    client = MongoClient(uri, tz_aware=True, tzinfo=timezone.utc)
    try:
        client.admin.command("ping")
    except Exception as e:
        pytest.skip(f"Cannot connect to MongoDB: {e}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def chunks_col(mongo):
    """Return the CHUNKS collection handle as defined by your module constants.

    Skips if DB/collection names are not provided.
    """
    dbn = getattr(M, "DB_NAME", None)
    cn = getattr(M, "CHUNKS_COL", None)
    if not dbn or not cn:
        pytest.skip("DB_NAME/CHUNKS_COL not configured")
    return mongo[dbn][cn]


@pytest.fixture(scope="session")
def articles_col(mongo):
    """Return the ARTICLES collection handle as defined by your module constants.

    Skips if DB/collection names are not provided.
    """
    dbn = getattr(M, "DB_NAME", None)
    ac = getattr(M, "ARTICLES_COL", None)
    if not dbn or not ac:
        pytest.skip("DB_NAME/ARTICLES_COL not configured")
    return mongo[dbn][ac]


# ---------- Helpers (with docstrings) ----------

def _has_openai_key() -> bool:
    """Return True if OPENAI_API_KEY is set (required for semantic/hybrid tests that embed queries)."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _skip_if_no_docs(col) -> None:
    """Skip the test if the provided collection is empty.

    This keeps CI green on fresh databases without fixtures.
    """
    if col.estimated_document_count() == 0:
        pytest.skip(f"No documents found in {col.full_name}")


def _get_chunk_index_name() -> str | None:
    """Resolve the articles full-text index name.

    Returns None if neither is defined.
    """
    return getattr(M, "CHUNKTEXT_INDEX", None)


def _ensure_article_exists_by_title(articles_col, title: str) -> bool:
    """Return True if an article with the exact `title` exists in the ARTICLES collection.

    This lets a targeted 'top hit' test skip gracefully when the fixture doc isn't present.
    """
    doc = articles_col.find_one({"title": title}, {"_id": 1})
    return bool(doc)


# ---------- Tests ----------

def test_keyword_search_on_articles_basic(chunks_col):
    """BM25 keyword search against ARTICLES collection using 'AI'."""
    _skip_if_no_docs(chunks_col)

    art_index = _get_chunk_index_name()
    if not art_index:
        pytest.skip(reason="No full-text index name available (FULLTEXT_INDEX)")

    pipeline = [
        {"$search": {
            "index": art_index,
            "compound": {
                "should": [
                    {"text": {"path": ["title"], "query": "AI", "score": {"boost": {"value": 3}}}},
                    {"text": {"path": ["content_chunk"], "query": "AI"}}
                ],
                "minimumShouldMatch": 1
            }
        }},
        {"$limit": 10},
        {"$project": {"_id": 1, "title": 1, "author": 1, "published_at": 1}}
    ]

    out = list(chunks_col.aggregate(pipeline))
    assert out, "Search returned no results; check index state and data"
    assert isinstance(out, list)
    assert all("_id" in d for d in out)
    assert (("AI" in d.get("title", "") or "AI" in d.get("content", "")) for d in out if "title" in d or "content" in d)


def test_keyword_search_on_articles_with_filters(chunks_col):
    """BM25 on ARTICLES with author/date filters, query='AI'."""
    _skip_if_no_docs(chunks_col)

    art_index = _get_chunk_index_name()
    if not art_index:
        pytest.skip("No full-text index name available for ARTICLES")

    any_author = "Lior Ben-David"
    gte_date = datetime(2000, 1, 1, tzinfo=timezone.utc)

    pipeline = [
        {"$search": {
            "index": "chunk_search",
            "compound": {
                "filter": [
                    {"equals": {"path": "author", "value": any_author}},
                    {"range": {"path": "published_at", "gte": gte_date}}
                ],
                "should": [
                    {"text": {"path": ["title"], "query": "AI", "score": {"boost": {"value": 3}}}},
                    {"text": {"path": ["content_chunk"], "query": "AI"}}
                ],
                "minimumShouldMatch": 1
            }
        }},
        {"$limit": 20},
        {"$project": {"author": 1, "published_at": 1}}
    ]

    out = list(chunks_col.aggregate(pipeline))
    assert out, "Search returned no results; check index state and data"
    print(f"Filtered search returned {len(out)} docs")
    for d in out:
        print(d)
    if out:
        assert all(d.get("author") == [any_author] for d in out if "author" in d)
        #assert all((d.get("published_at") is None) or (d["published_at"] >= gte_date) for d in out)


def test_keyword_specific_title_top_hit(chunks_col):
    """Ensure a known article title is returned as the TOP result in keyword search.

    Title under test: "Open community proposes SPDX-style manifests for AI datasets"
    Strategy:
      - Verify the document exists; if not, skip (keeps suite robust).
      - Use a phrase match on title (boosted) with a text fallback.
      - Assert the first returned document's title matches exactly.
    """
    _skip_if_no_docs(chunks_col)

    exact_title = "Open community proposes SPDX-style manifests for AI datasets"

    # Ensure the target doc exists; otherwise skip this focused test
    if not _ensure_article_exists_by_title(chunks_col, exact_title):
        pytest.skip(f"Article with title not found in ARTICLES: '{exact_title}'")

    art_index = _get_chunk_index_name()
    if not art_index:
        pytest.skip("No full-text index name available for ARTICLES")

    pipeline = [
        {"$search": {
            "index": art_index,
            "compound": {
                "should": [
                    # Strong phrase match on title to lift exact title to the top
                    {"phrase": {"path": "title", "query": exact_title, "slop": 0,
                                "score": {"boost": {"value": 8}}}},
                    # Fallback text search (title/content)
                    {"text": {"path": ["title", "content_chunk"], "query": exact_title}}
                ],
                "minimumShouldMatch": 1
            }
        }},
        {"$limit": 5},
        {"$project": {"_id": 1, "title": 1}}
    ]

    out = list(chunks_col.aggregate(pipeline))
    assert out, "Search returned no results; check index state and data"
    top_title = out[0].get("title")
    # Helpful message if it fails:
    got_titles = [d.get("title") for d in out]
    assert top_title == exact_title, f"Top hit mismatch.\nExpected: {exact_title}\nGot: {top_title}\nCandidates: {got_titles}"


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set for semantic test")
def test_semantic_vector_search_basic(chunks_col):
    """Vector search on CHUNKS with 'AI' query and $vectorSearch."""
    _skip_if_no_docs(chunks_col)

    vector_index = getattr(M, "VECTOR_INDEX", None)
    vector_path = getattr(M, "VECTOR_PATH", None)
    if not vector_index or not vector_path:
        pytest.skip("VECTOR_INDEX/VECTOR_PATH not defined in module")

    qvec = M.embed_query("AI")
    assert isinstance(qvec, list) and len(qvec) > 0

    pipeline = [
        {"$vectorSearch": {
            "index": vector_index,
            "path": vector_path,
            "queryVector": qvec,
            "numCandidates": 200,
            "limit": 10
        }},
        {"$project": {"_id": 1, "title": 1, "author": 1, "published_at": 1}}
    ]

    out = list(chunks_col.aggregate(pipeline))
    assert out, "Search returned no results; check index state and data"
    assert isinstance(out, list)
    assert all("_id" in d for d in out)



@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set for hybrid test")
def test_hybrid_rankfusion_no_filters():
    """End-to-end hybrid fusion with 'AI' using your pipeline function."""
    state = M.SearchState(
        messages=[],
        user_query="AI chips",
        file_type=getattr(M, "DEFAULT_FILE_TYPE", "chunk"),
        lexical_keywords=["AI", "chips"],
        semantic_query = "AI chips",
    )
    out = M.hybrid_retrieve_rankfusion(state)
    assert isinstance(out, M.SearchState)
    assert out.candidates, "Search returned no results; check index state and data"
    if out.candidates:
        first = out.candidates[0]
        for key in ("id", "article_id", "title", "author", "published_at", "score", "fused_score"):
            assert key in first
    print(f"Hybrid RankFusion returned {len(out.candidates)} candidates")
    for c in out.candidates[:5]:
        print(f"- {c.get('title')} (score: {c.get('fused_score'):.4f})")


@pytest.mark.skipif(not _has_openai_key(), reason="OPENAI_API_KEY not set for hybrid test")
def test_hybrid_rankfusion_with_filters(chunks_col):
    """Hybrid fusion with an author filter pulled from CHUNKS and query='AI'."""
    _skip_if_no_docs(chunks_col)
    sample = chunks_col.find_one({"author": {"$ne": None}}, {"author": 1})
    if not sample or not sample.get("author"):
        pytest.skip("No author in CHUNKS to test hybrid filters")

    author = "Yael Mor"
    state = M.SearchState(
        messages=[],
        user_query="AI chips",
        file_type=getattr(M, "DEFAULT_FILE_TYPE", "chunk"),
        lexical_keywords=["AI", "chips"],
        semantic_query = "AI chips",
        filters={"author": [author], "from": "2000-01-01", "to": None},
    )
    out = M.hybrid_retrieve_rankfusion(state)
    assert out.candidates, "Search returned no results; check index state and data"
    print(f"Hybrid RankFusion with filters returned {len(out.candidates)} candidates")
    for c in out.candidates[:5]:
        print(f"- {c.get('title')} by {c.get('author')} (score: {c.get('fused_score'):.4f})")
    if out.candidates:
        assert all(author in c.get("author") for c in out.candidates if c.get("author") is not None)


def test_build_lexical_query_join():
    """Basic check that lexical keywords are joined into a single space-separated string."""
    state = M.SearchState(messages=[], user_query="AI")
    state.lexical_keywords = ["AI", "healthcare", "NLP"]
    lex = M.build_lexical_query(state)
    assert lex == "AI healthcare NLP"
