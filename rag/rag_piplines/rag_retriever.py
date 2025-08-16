# retriever.py
import os, pickle, faiss, numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")


class RAGRetriever:
    def __init__(self, index_and_metadata_base_path="./rag/data_indexing/indexes_and_metadata_files"):
        # Load all 4 indexes + metadata

        # article + auther + content index
        self.full_content_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "full_article_content.index"))
        self.full_content_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "full_article_content_metadata.pkl"), "rb"))

        # article index
        self.article_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "articles.index"))
        self.article_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "articles_metadata.pkl"), "rb"))

        # article chunks index
        self.chunk_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "chunks.index"))
        self.chunk_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "chunks_metadata.pkl"), "rb"))

        # article titles index
        self.title_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "titles.index"))
        self.title_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "titles_metadata.pkl"), "rb"))

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # ----- connect Mongo
        self._mongo = MongoClient(MONGO_URI)[DB][COLL]

    def _query_embed(self, q: str) -> np.ndarray:
        """
        This function generates an embedding for the given query using OpenAI's API.
        Args:
            q (str): The query to generate an embedding for.
        Returns:
            np.ndarray: The generated embedding as a NumPy array with size (1536,) ready for FAISS indexing.
        """
        v = self._client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding
        return np.array(v, dtype=np.float32).reshape(1, -1)

    def _l2_to_sim(self, d: float) -> float:
        """
        Convert an L2 (Euclidean) distance into a similarity score between 0 and 1.

        Args:
            d (float): The L2 distance between two vectors (smaller = more similar).

        Returns:
            float: Similarity score where higher means more similar.

        Notes:
            - FAISS with IndexFlatL2 returns distances; we invert them into a similarity metric
            for easier scoring.
            - Formula: similarity = 1 / (1 + distance)
            - This is simple; other transformations (e.g., negative distance) are also possible.
        """
        return 1.0 / (1.0 + d)


    def _parse_dt(self, s: str):
        """
        Safely parse a date/time string into a Python datetime object.

        Args:
            s (str): A date/time string (ISO8601, RFC2822, etc.), or None.

        Returns:
            datetime or None: Parsed datetime (timezone-aware if input has timezone info),
            or None if parsing fails.

        Notes:
            - Uses `dateutil.parser.parse` which supports many date formats.
            - Returns None if parsing fails or if input is None.
        """
        try:
            return dtparser.parse(s) if s else None
        except:
            return None


    def _recency_weight(self, dt, half_life_days=30) -> float:
        """
        Compute a 'freshness score' for a document based on how recent it is.

        Args:
            dt (datetime): The publication date/time of the document.
            half_life_days (int, optional): The number of days for the score to decay by half.
                                            Defaults to 30.

        Returns:
            float: Recency weight between 0 and 1, higher = more recent.

        Notes:
            - If `dt` is None, return a default penalty (0.6).
            - The formula is exponential decay: weight = 0.5 ^ (age_days / half_life_days)
            This means:
                * age_days = 0 → weight = 1.0 (fresh)
                * age_days = half_life_days → weight = 0.5
                * age_days = 2 × half_life_days → weight = 0.25
            - This allows newer docs to be ranked higher even if their semantic similarity
            is slightly lower.
        """
        if not dt:
            return 0.6  # penalize but don't eliminate - Maybe use a different value?
        now = datetime.now(timezone.utc)
        doc_age = max(0, (now - dt).days)
        return 0.5 ** (doc_age / float(half_life_days)) # 2^(doc_age / half_life_days) - ^ = ** in Python
    
    def normalize_metadata(self, meta: dict) -> dict:
        """Return a uniform dict for any index."""
        out = {
            "title": meta.get("title", None),
            "url": meta.get("url", None),
            "content": meta.get("content", None),                          
            "author": meta.get("author", None),
            "published_at": meta.get("published_at", None),
            "chunk": meta.get("chunk", None), 
            "chunk_id": meta.get("chunk_id", None),                 
        }

        return out

    def _semantic_search(self, index_file, meta_file, user_query, k_semantic_matches=20):

        """
        Search a given FAISS index_file and return the top results, re-ranked by recency.

        Args:
            index_file (faiss.Index): The FAISS index to search (article-level, chunk-level, etc.).
            meta (list[dict]): Metadata list where meta[i] corresponds to index vector i.
                            Each dict should have at least:
                                - "published_at" (ISO string or None)
                                - "title", "url", and content/chunk text
            query (str): The user query to search for.
            k_initial_matches (int): How many top matches should we ask FAISS for.
            k_final_matches (int): Number of top results to return after re-ranking.

        Returns:
            list[tuple]: A list of up to k_final tuples:
                (final_score, index_id, similarity, recency_weight, metadata_dict)
        """

        # 1. Convert query text → vector
        query_vector = self._query_embed(user_query)

        # 2. Search FAISS index: get distances + vector IDs
        dists, ids = index_file.search(query_vector, k_semantic_matches) # return the k_initial_matches nearest ones using your distance metric (L2 in this case)
        # dists - the distances to the k_initial_matches nearest neighbors
        # ids - the IDs of the k_initial_matches nearest neighbors
        dists, ids = dists[0], ids[0]  # FAISS returns nested arrays [[...]] ; take first row

        cands = []

        # 3. Iterate over search results
        for dist, i in zip(dists, ids):
            print(f"Processing vector ID {i} with distance {dist:.4f}\n\n")
            if i < 0:  # invalid ID (padding from FAISS if fewer than k_initial_matches results)
                continue

            # 4. Get metadata for this vector
            metadata_i = meta_file[i]

            # 5. Convert distance to similarity (higher = more similar)
            semantic_score = self._l2_to_sim(dist)

            # 6. Normalize metadata for consistent output
            metadata_i = self.normalize_metadata(metadata_i)
            metadata_i["semantic_score"] = semantic_score
            
            # 7. Add to candidate list
            cands.append(metadata_i)

        # 8. Sort candidates by semantic_score (descending)
        cands.sort(key=lambda x: x["semantic_score"], reverse=True)

        return cands

    def _keyword_search(self, user_query, fields=["title", "author", "content"], k_keyword_matches=20):
        """
        Perform a keyword search in MongoDB using Atlas Search.

        Returns a normalized list of results with uniform fields.
        """
        if not user_query.strip():
            print("Skipping keyword search because query is empty.")
            return []
    
        pipeline = [
            {
                "$search": {
                    "index": "article_search",
                    "text": {
                        "query": user_query,
                        "path": fields,
                        "fuzzy": {"maxEdits": 1}
                    }
                }
            },
            {
                "$project": {
                    "title": 1,
                    "author": 1,
                    "content": 1,
                    "url": 1,
                    "published_at": 1,
                    "bm25_score": {"$meta": "searchScore"}
                }
            },
            {"$limit": int(k_keyword_matches)},
            {"$sort": {"bm25_score": -1}}
        ]

        raw_results = list(self._mongo.aggregate(pipeline))

        # Normalize for consistency with semantic results
        normalized_results = []
        for doc in raw_results:
            norm = self.normalize_metadata(doc)
            norm["bm25_score"] = doc.get("bm25_score", 0.0)
            normalized_results.append(norm)

        return normalized_results


    def _merge_and_score_searches(self, semantic_results, keyword_results, alpha=0.7, beta=0.3):
        """
        Merge semantic and keyword results.
        - Keeps multiple results per article if chunks differ.
        - Computes a weighted final score.

        Args:
            semantic_results (list): Each item may be an article or a chunk.
            keyword_results (list): Full-article matches from keyword search.
            alpha (float): Semantic score weight.
            beta (float): Keyword score weight.

        Returns:
            list: Scored, sorted list of documents/chunks.
        """
        index = {}  # key = (url, chunk_id or None)

        def make_key(doc):
            return (doc.get("url"), doc.get("chunk_id"))  # allows multiple chunks per article

        # Add semantic results
        for doc in semantic_results:
            key = make_key(doc)
            index[key] = dict(doc)  # full copy
            index[key]["semantic_score"] = doc.get("semantic_score", 0.0)

        # Add keyword results
        for doc in keyword_results:
            key = make_key(doc)
            if key in index:
                index[key].update(doc)
            else:
                index[key] = dict(doc)
            index[key]["bm25_score"] = doc.get("bm25_score", 0.0)

        # Compute final weighted score
        scored = []
        for doc in index.values():
            s = doc.get("semantic_score", 0.0)
            b = doc.get("bm25_score", 0.0)
            doc["final_score"] = alpha * s + beta * b
            scored.append(doc)

        # Sort by final_score
        scored.sort(key=lambda d: d["final_score"], reverse=True)
        return scored


    def _hybrid_search(self, query, 
                       index_file, 
                       meta_file, 
                       keyword_fields=["title", "author", "content"],
                       alpha=0.7, 
                       beta=0.3,
                       k_semantic_matches=30, 
                       k_keyword_matches=30, 
                       k_final=6):
        """
        This function performs a hybrid search, combining semantic and keyword search.
        Args:
            query (str): The query string to search for.
            index_file (faiss.Index): The FAISS index to search.
            meta_file (list): The metadata list where meta[i] corresponds to index vector i.
            keyword_fields (list): The fields to search in for keyword search.
            alpha (float): The weight of the semantic score in the final score.
            beta (float): The weight of the keyword score in the final score.
            k_semantic (int): The number of semantic search results to return.
            k_keyword (int): The number of keyword search results to return.
            k_final (int): The maximum number of final search results to return.
        Returns:
            list: A list of documents matching the search query.
        """
        if alpha > 0:
            semantic_results = self._semantic_search(index_file=index_file,
                                                meta_file=meta_file,
                                                user_query=query,
                                                k_semantic_matches=k_semantic_matches)
        else:
            semantic_results = []

        if beta > 0:
            keyword_results = self._keyword_search(user_query=query,fields=keyword_fields, k_keyword_matches=k_keyword_matches)

        else:
            keyword_results = []

        merged = self._merge_and_score_searches(semantic_results=semantic_results, 
                                       keyword_results=keyword_results, 
                                       alpha=alpha, 
                                       beta=beta)

        return merged[:k_final]



    def retrieve(self, 
                 query, 
                 semantic_file="full_content", 
                 keywords_fields=["title", "author", "content"], 
                 alpha=0.7, 
                 beta=0.3,
                 k_semantic_matches=20, 
                 k_keyword_matches=20,
                 k_final_matches=6,):
        """
        Perform a hybrid search based on the selected mode: 'article', 'chunk', or 'title'.

        Args:
            query (str): User query.
            semantic_file (str): The type of semantic index to use ('full_content', 'article', 'chunk', 'title').
            keywords_fields (list): Fields to use for keyword search (Can be 'title', 'author', 'content').
            alpha (float): Weight for semantic score in final ranking.
            beta (float): Weight for keyword score in final ranking.
            mode (str): Search mode - 'article', 'chunk', or 'title'.
            k_semantic_matches (int): Number of initial matches to retrieve from semantic search.
            k_keyword_matches (int): Number of initial matches to retrieve from keyword search.
            k_final_matches (int): Number of final matches to return after merging and scoring.

        Returns:
            list: Top-k results sorted by combined relevance score.

        """
        if semantic_file == "full_content":
            return self._hybrid_search(
                query=query,
                index_file=self.full_content_idx,
                meta_file=self.full_content_meta,
                keyword_fields=keywords_fields,
                alpha=alpha,
                beta=beta,
                k_semantic_matches=k_semantic_matches,
                k_keyword_matches=k_keyword_matches,
                k_final=k_final_matches
            )
        if semantic_file == "article":
            return self._hybrid_search(
                query=query,
                index_file=self.article_idx,
                meta_file=self.article_meta,
                keyword_fields=keywords_fields,
                alpha=alpha,
                beta=beta,
                k_semantic_matches=k_semantic_matches,
                k_keyword_matches=k_keyword_matches,
                k_final=k_final_matches
            )

        if semantic_file == "chunk":
            return self._hybrid_search(
                query=query,
                index_file=self.chunk_idx,
                meta_file=self.chunk_meta,
                keyword_fields=keywords_fields,
                alpha=1,
                beta=0,
                k_semantic_matches=k_semantic_matches,
                k_keyword_matches=k_keyword_matches,
                k_final=k_final_matches
            )

        if semantic_file == "title":
            return self._hybrid_search(
                query=query,
                index_file=self.title_idx,
                meta_file=self.title_meta,
                keyword_fields=keywords_fields,
                alpha=alpha,
                beta=beta,
                k_semantic_matches=k_semantic_matches,
                k_keyword_matches=k_keyword_matches,
                k_final=k_final_matches
            )

        raise ValueError(f"Unknown retrieval mode: {semantic_file}")


  