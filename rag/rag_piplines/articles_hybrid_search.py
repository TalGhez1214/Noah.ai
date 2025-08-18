
from collections import defaultdict
from copy import deepcopy
from rag.data_indexing.chuncks_indexing import embed, l2_to_sim
from pymongo import MongoClient

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")

# ----- connect Mongo
mongo = MongoClient(MONGO_URI)[DB][COLL]

def keyword_search(query, limit=20):
    pipeline = [
        {
            "$search": {
                "index": "article_search",
                "text": {
                    "query": query,
                    "path": ["title", "author", "content"],
                    "fuzzy": { "maxEdits": 1 }
                }
            }
        },
        { "$limit": limit },
        { "$project": {
            "title": 1,
            "author": 1,
            "content": 1,
            "url": 1,
            "published_at": 1,
            "bm25_score": { "$meta": "searchScore" }
        }}
    ]
    return list(mongo.articles.aggregate(pipeline))



def merge_and_score(semantic_results, keyword_results, query, alpha=0.7, beta=0.3):
    index = defaultdict(dict)

    # Add semantic matches
    for doc in semantic_results:
        doc_id = doc.get("url")  # use URL or Mongo `_id` as unique key
        index[doc_id].update(doc)
        index[doc_id]["semantic_score"] = doc.get("semantic_score", 0.0)

    # Add keyword matches
    for doc in keyword_results:
        doc_id = doc.get("url")
        index[doc_id].update(doc)
        index[doc_id]["bm25_score"] = doc.get("bm25_score", 0.0)

    # Final score = alpha * semantic + beta * keyword
    scored = []
    for doc_id, doc in index.items():
        s = doc.get("semantic_score", 0.0)
        b = doc.get("bm25_score", 0.0)
        doc["final_score"] = alpha * s + beta * b
        scored.append(doc)

    scored.sort(key=lambda d: d["final_score"], reverse=True)
    return scored



def hybrid_search(query, k_semantic=30, k_keyword=30, final_k=5):
    # Step 1: semantic via FAISS
    query_vector = embed(query)
    dists, ids = faiss_index.search(query_vector, k_semantic)
    semantic_results = [
        { **deepcopy(meta_file[i]), "semantic_score": l2_to_sim(d) }
        for d, i in zip(dists[0], ids[0]) if i >= 0
    ]

    # Step 2: keyword via MongoDB Atlas
    keyword_results = keyword_search(query, limit=k_keyword)

    # Step 3: merge + rerank
    merged = merge_and_score(semantic_results, keyword_results, query)
    return merged[:final_k]
