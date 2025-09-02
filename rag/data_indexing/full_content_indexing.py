import os
import pickle
import numpy as np
import faiss
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

"""
This script indexes full articles conent - title, auther, content - from a MongoDB collection into a FAISS index.
"""


# -------------------- MongoDB Setup --------------------
client_mongo = MongoClient(os.getenv("MONGODB_URI"))
db = client_mongo[os.getenv("DB_NAME", "news")]
col = db[os.getenv("COLLECTION_NAME", "toi_articles")]

# -------------------- OpenAI Setup --------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- Step 1: Fetch Articles --------------------
docs = list(col.find(
    {},
    {
        "_id": 0,
        "title": 1,
        "author": 1,
        "content": 1,
        "url": 1,
        "published_at": 1
    }
))

# -------------------- Step 2: Embed Full Article (Title + Author + Content) --------------------
def embed_text(text: str) -> np.ndarray:
    """
    Generate a semantic embedding for the given text using OpenAI's API.

    Args:
        text (str): The text to embed.

    Returns:
        np.ndarray: The embedding vector (shape: 1 x 1536).
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# -------------------- Step 3: Generate Embeddings + Metadata --------------------
embeddings = []
metadata = []

for doc in docs:
    if not doc.get("title") or not doc.get("content") or not doc.get("author"):
        continue  # Skip incomplete articles

    full_text = (
    f"Article Title: {doc['title']}\n"
    f"Article Author: {doc['author']}\n"
    f"Article Content: {doc['content']}"
    )
    vector = embed_text(full_text)

    embeddings.append(vector)
    metadata.append(doc)

embeddings_np = np.vstack(embeddings)

# -------------------- Step 4: Build FAISS Index --------------------
dimension = embeddings_np.shape[1]  # typically 1536
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# -------------------- Step 5: Save Index and Metadata --------------------
faiss.write_index(index, "./rag/data_indexing/indexes_and_metadata_files/full_article_content.index")

with open("./rag/data_indexing/indexes_and_metadata_files/full_article_content_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Indexed {len(metadata)} articles with full semantic embeddings.")
