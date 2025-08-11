# retriever.py
import os, pickle, faiss, numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGRetriever:
    def __init__(self, index_and_metadata_base_path="./rag/data_indexing/indexes_and_metadata_files"):
        # Load all three indexes + metadata
        self.article_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "articles.index"))
        self.article_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "articles_metadata.pkl"), "rb"))

        self.chunk_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "chunks.index"))
        self.chunk_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "chunks_metadata.pkl"), "rb"))

        self.title_idx = faiss.read_index(os.path.join(index_and_metadata_base_path, "titles.index"))
        self.title_meta = pickle.load(open(os.path.join(index_and_metadata_base_path, "titles_metadata.pkl"), "rb"))

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    def _search_index(self, index_file, meta_file, user_query, k_initial_matches, k_final_matches, use_recency_weight = True,half_life_days = 30, min_days_window=None):

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
            half_life_days (int): Recency decay parameter. Lower = faster decay in score.
            min_days_window (int or None): Optional cutoff; discard results older than this
                                        many days. If None, keep all.

        Returns:
            list[tuple]: A list of up to k_final tuples:
                (final_score, index_id, similarity, recency_weight, metadata_dict)

        Processing steps:
            1. **Embed the query** into a vector (using `_embed`).
            2. **Search FAISS** for top-k_search nearest vectors (L2 distance).
            3. **Convert distances to similarity** with `_l2_to_sim`.
            4. **Parse publication date** from metadata with `_parse_dt`.
            5. If `min_days_window` is set, **skip old results** beyond that window.
            6. **Compute recency weight** with `_recency_weight`.
            7. **Final score** = semantic similarity × recency weight.
            8. Collect all candidates into a list.
            9. **Sort candidates** by final score (highest first).
            10. Return only the top k_final results.
        """

        # 1. Convert query text → vector
        query_vector = self._query_embed(user_query)

        # 2. Search FAISS index: get distances + vector IDs
        dists, ids = index_file.search(query_vector, k_initial_matches) # return the k_initial_matches nearest ones using your distance metric (L2 in this case)
        # dists - the distances to the k_initial_matches nearest neighbors
        # ids - the IDs of the k_initial_matches nearest neighbors
        dists, ids = dists[0], ids[0]  # FAISS returns nested arrays [[...]] ; take first row

        now = datetime.now(timezone.utc)
        cands = []

        # 3. Iterate over search results
        for dist, i in zip(dists, ids):
            if i < 0:  # invalid ID (padding from FAISS if fewer than k_initial_matches results)
                continue

            # 4. Get metadata for this vector
            m = meta_file[i]

            # 5. Convert distance to similarity (higher = more similar)
            sim = self._l2_to_sim(dist)

            # 6. Parse publication date
            dt = self._parse_dt(m.get("published_at"))

            # 7. If cutoff window is set, skip too-old items
            if min_days_window is not None and dt is not None:
                if (now - dt).days > min_days_window:
                    continue

            # 8. Compute recency weight (boost newer results)
            if use_recency_weight:
                rw = self._recency_weight(dt, half_life_days=half_life_days)
            else:
                rw = 1.0

            # 9. Final score = semantic similarity * recency weight
            final_score = sim * rw

            # 10. Add to candidate list
            cands.append((final_score, i, sim, rw, m))

        # 11. Sort candidates by final score (descending)
        cands.sort(key=lambda x: x[0], reverse=True)

        # 12. Return top-k_final_matches results
        return cands[:k_final_matches]

    def retrieve(self, question, mode="auto", k_initial_matches=80, k_final_matches=6):
        if mode == "article":
            return self._search_index(index_file = self.article_idx, 
                                      meta_file = self.article_meta, 
                                      user_query = question,
                                      k_initial_matches=k_initial_matches,
                                        k_final_matches=k_final_matches,
                                      )
        if mode == "chunk":
            return self._search_index(index_file = self.chunk_idx, 
                                      meta_file = self.chunk_meta, 
                                      user_query = question,
                                      k_initial_matches=k_initial_matches,
                                      k_final_matches=k_final_matches,
                                      )
        if mode == "title":
            return self._search_index(index_file = self.title_idx, 
                                      meta_file = self.title_meta, 
                                      user_query = question,
                                      k_initial_matches=k_initial_matches,
                                      k_final_matches=k_final_matches,
                                      )


    