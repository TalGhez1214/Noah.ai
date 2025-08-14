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

def test_chunk_metadata_file():
    """
    Test the chunk metadata file.

    Reads the chunks_metadata.pkl file and prints the chunks in the order of the chunks.index file.
    """
    import pickle
    import faiss

    # Load the chunk index
    index = faiss.read_index("./rag/data_indexing/indexes_and_metadata_files/chunks.index")

    # Load the chunk metadata
    with open("./rag/data_indexing/indexes_and_metadata_files/chunks_metadata.pkl", "rb") as f:
        chunk_meta = pickle.load(f)

    # Get the chunk IDs in the order of the index
    chunk_ids = index.ntotal

    # Print the chunks in the order of the index
    for chunk_id in range(chunk_ids):
        chunk = chunk_meta[chunk_id]
        print(f"  Title: {chunk['title']}")
        print(f"  URL: {chunk['url']}")
        print(f"  Published at: {chunk['published_at']}")
        print(f"  Author: {chunk['author']}")
        print(f"  Chunk ID: {chunk['chunk_id']}\n\n")
        print(f"  Chunk: {chunk['chunk']}")
        print("#" * 80)

def test_article_search_index(rag_retriever):
    """
    Test the search index function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Who is Benjamin Netanyahu?"
    results = rag_retriever._search_index(
        index_file=rag_retriever.article_idx,
        meta_file=rag_retriever.article_meta,
        user_query=query,
        k_initial_matches=5,
        k_final_matches=2,
    )
    assert isinstance(results, list)
    assert len(results) == 2

def test_article_retrieve(rag_retriever):
    """
    Test the retrieve function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "Who is Benjamin Netanyahu?"
    results = rag_retriever.retrieve(query, mode="article", k_final_matches=1)
    assert isinstance(results, list)
    assert len(results) == 1

def test_retrieve_chunk(rag_retriever):
    """
    Test the retrieve chunk function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "tell me about Yeshiva students?"
    results = rag_retriever.retrieve(query, mode="chunk", k_initial_matches=4,k_final_matches=4)
    assert isinstance(results, list)
    assert len(results) <= 6

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

def test_retrieve_title(rag_retriever):
    """
    Test the retrieve title function.

    Args:
        rag_retriever (RAGRetriever): An instance of the RAGRetriever class.

    Returns:
        None
    """
    query = "tell me about the crisis in Gaza?"
    results = rag_retriever.retrieve(query, mode="title", k_final_matches=1)
    assert isinstance(results, list)
    assert len(results) == 1

    print(f"Retrieved {len(results)} titles for query '{query}':\n\n")
    for res in results:
        print(f"Score: {res["final_score"]}\n\n")
        print(f"Title: {res['title']}\n")
        print("-" * 80)