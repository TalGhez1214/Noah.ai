# build_chunked_index.py
import os, math, pickle
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import faiss
from openai import OpenAI
import tiktoken  # pip install tiktoken
from dateutil import parser as dtparser  # pip install python-dateutil

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MONGO_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "news")
COLL = os.getenv("COLLECTION_NAME", "toi_articles")

# ----- connect Mongo
col = MongoClient(MONGO_URI)[DB][COLL]

# ----- tokenizer (gpt-4o/gpt-3.5 compatible)
enc = tiktoken.get_encoding("cl100k_base")

def embed(text: str) -> np.ndarray:
    # Embedding for a single chunk
    v = client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    return np.array(v, dtype=np.float32)

# ----- fetch docs
docs = list(col.find({}, {"_id": 0, "title": 1, "url": 1, "content": 1, "published_at": 1, "author": 1}))

embeds = []
meta = []

for doc in docs:
    title = (doc.get("title") or "").strip()
    if not title:
        continue
    vec = embed(title)
    embeds.append(vec)
    meta.append(doc)

if not embeds:
    raise SystemExit("No chunks to index. Check your Mongo collection content.")

X = np.vstack(embeds)
dim = X.shape[1]

# simple, exact index (great up to ~100k vectors)
index = faiss.IndexFlatL2(dim)
index.add(X)

faiss.write_index(index, "./rag/data_indexing/indexes_and_metadata_files/titles.index")
with open("./rag/data_indexing/indexes_and_metadata_files/titles_metadata.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"Stored {len(meta)} titles in FAISS index.")
