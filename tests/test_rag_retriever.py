"""
Test file for the RAGRetriever class.

This file contains unit tests for the RAGRetriever class, which is responsible for
retrieving relevant documents from a search index.

"""

import pytest
from rag.rag_piplines.rag_retriever import RAGRetriever
import numpy as np
from datetime import datetime, timezone
import faiss
import pickle

@pytest.fixture
def rag_retriever():
    """
    Fixture to create a RAGRetriever instance.

    Returns:
        RAGRetriever: An instance of the RAGRetriever class.
    """
    base_path = "./rag/data_indexing/indexes_and_metadata_files"
    return RAGRetriever(base_path)

def test_rag_retriever_init(rag_retriever):
    """
    Test the initialization of the RAGRetriever class.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    assert hasattr(rag_retriever, "full_content_idx")
    assert hasattr(rag_retriever, "full_content_meta")
    assert hasattr(rag_retriever, "article_idx")
    assert hasattr(rag_retriever, "article_meta")
    assert hasattr(rag_retriever, "chunk_idx")
    assert hasattr(rag_retriever, "chunk_meta")
    assert hasattr(rag_retriever, "title_idx")
    assert hasattr(rag_retriever, "title_meta")

def test_query_embed(rag_retriever):
    """
    Test the query embedding function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "test query"
    embedding = rag_retriever._query_embed(query)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 1536)

def test_parse_dt(rag_retriever):
    """
    Test the date parsing function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    dt_str = "2022-01-01T00:00:00Z"
    dt = rag_retriever._parse_dt(dt_str)
    assert isinstance(dt, datetime)
    assert dt.astimezone(timezone.utc) == dt

def test_recency_weight(rag_retriever):
    """
    Test the recency weight calculation function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    dt = datetime(2025, 8, 1, tzinfo=timezone.utc)
    weight = rag_retriever._recency_weight(dt)
    assert isinstance(weight, float)
    assert 0 <= weight <= 1

    print(f"Recency weight for {dt}: {weight}")

def test_l2_to_sim(rag_retriever):
    """
    Test the similarity calculation function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    dist = 1.0
    sim = rag_retriever._l2_to_sim(dist)
    assert isinstance(sim, float)
    assert 0 <= sim <= 1


def test_article_retrieve(rag_retriever):
    """
    Test the retrieve function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Who is Benjamin Netanyahu?"
    results = rag_retriever.retrieve(query, semantic_file="article", k_final_matches=1)
    assert isinstance(results, list)
    assert len(results) == 1


def test_article_semantic_search(rag_retriever):
    """
    Test the search index function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Who is Benjamin Netanyahu?"
    results = rag_retriever._semantic_search(
        user_query=query,
        index_file=rag_retriever.article_idx,
        meta_file=rag_retriever.article_meta,
        k_semantic_matches=4,
    )
    assert isinstance(results, list)
    assert len(results) == 4

def test_full_content_semantic_search(rag_retriever):
    """
    Test the search index function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "give me information about Armenia and Azerbaijan?"
    results = rag_retriever._semantic_search(
        user_query=query,
        index_file=rag_retriever.full_content_idx,
        meta_file=rag_retriever.full_content_meta,
        k_semantic_matches=4,
    )
    assert isinstance(results, list)
    assert len(results) == 4

def test_keyword_search(rag_retriever):
    """
    Test the keyword search function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Palestinian"
    k_keyword_matches = 3
    results = rag_retriever._keyword_search(query, k_keyword_matches=k_keyword_matches)
    assert isinstance(results, list)
    assert len(results) <= k_keyword_matches

    print(f"Retrieved {len(results)} results for query '{query}':\n\n")
    for res in results:
        print(f"BM25 Score: {res['bm25_score']}\n\n")
        print(f"Title: {res['title']}\n")
        print(f"Author: {res['author']}\n")

        print("-" * 80)

def test_title_keyword_search(rag_retriever):
    """
    Test the keyword search function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Azerbaijan"
    k_keyword_matches = 3
    results = rag_retriever._keyword_search(query,fields=["title"], k_keyword_matches=k_keyword_matches)
    assert isinstance(results, list)
    assert len(results) <= k_keyword_matches

    print(f"Retrieved {len(results)} results for query '{query}':\n\n")
    for res in results:
        print(f"BM25 Score: {res['bm25_score']}\n\n")
        print(f"Title: {res['title']}\n")
        print(f"Author: {res['author']}\n")

        print("-" * 80)

def test_retrieve_full_content(rag_retriever):
    """
    Test the retrieve full content function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "can you give me articles about Gaza of the auther Noam Harari?"
    results = rag_retriever.retrieve(query=query, 
                                     semantic_file="full_content",
                                     alpha=0.7,
                                     beta=0.3,
                                     k_final_matches=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2

    print(f"Retrieved {len(results)} doc for query '{query}':\n\n")

    seen = set()
    for i, res in enumerate(results, start=1):
        # make missing keys obvious but non-fatal
        sematic_score = res.get("semantic_score")
        bm25_score = res.get("bm25_score")
        score = res.get("final_score")
        title = res.get("title")
        auther = res.get("author")
        
        print(f"### Result {i} ###")
        print(f"Score: {score}")
        print(f"Semantic Score: {sematic_score}")
        print(f"BM25 Score: {bm25_score}")
        print(f"Title: {title}")
        print(f"Author: {auther}")
        print("-" * 80)

def test_retrieve_chunk(rag_retriever):
    """
    Test the retrieve chunk function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "can you give me articles about Gaza of the auther Noam Harari?"
    results = rag_retriever.retrieve(query=query, 
                                     semantic_file="chunk",
                                     k_final_matches=4)
    
    assert isinstance(results, list)
    assert len(results) <= 4

    print(f"Retrieved {len(results)} chunks for query '{query}':\n\n")

    seen = set()
    for i, res in enumerate(results, start=1):
        # make missing keys obvious but non-fatal
        score = res.get("final_score")
        chunk = res.get("chunk") 
        cid   = (res.get("url"), res.get("chunk_id"))

        print(f"### Chunk {i} ###")
        print(f"Score: {score}")
        print(f"Chunk ID: {res.get('chunk_id')}  URL: {res.get('url')}")
        print(f"Text: {chunk[:300]}...")
        print(f"Word count: {len((chunk or '').split())}")
        print("-" * 80)

        assert cid not in seen, "Duplicate chunk returned"
        seen.add(cid)







    