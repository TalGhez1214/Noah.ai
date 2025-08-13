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

def chunk_text(text, chunk_tokens=400, overlap_tokens=40):
    """Split text into overlapping token chunks.
    
    Args:
        text (str): The text to chunk.
        chunk_tokens (int): Number of tokens per chunk.
        overlap_tokens (int): Number of overlapping tokens between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    toks = enc.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + chunk_tokens, len(toks))
        chunk = enc.decode(toks[start:end])
        chunks.append(chunk)
        if end == len(toks):
            break
        start = end - overlap_tokens  # overlap
        if start < 0:
            start = 0
    return chunks

def embed(text: str) -> np.ndarray:
    # Embedding for a single chunk
    v = client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    return np.array(v, dtype=np.float32)

# ----- fetch docs
docs = list(col.find({}, {"_id": 0, "title": 1, "url": 1, "content": 1, "published_at": 1, "author": 1}))

embeds = []
meta = []

for doc in docs:
    content = (doc.get("content") or "").strip()
    if not content:
        continue
    chunks = chunk_text(content, chunk_tokens=400, overlap_tokens=40)
    for i, ch in enumerate(chunks):
        vec = embed(ch)
        embeds.append(vec)
        meta.append({
            "title": doc.get("title"),
            "url": doc.get("url"),
            "published_at": doc.get("published_at"),  # ISO string or None
            "author": doc.get("author"),
            "content": doc.get("content"),
            "chunk_id": i,
            "indexed_doc": ch
        })

if not embeds:
    raise SystemExit("No chunks to index. Check your Mongo collection content.")

X = np.vstack(embeds)
dim = X.shape[1]

# simple, exact index (great up to ~100k vectors)
index = faiss.IndexFlatL2(dim)
index.add(X)

faiss.write_index(index, "./rag/data_indexing//indexes_and_metadata_files/chunks.index")
with open("./rag/data_indexing//indexes_and_metadata_files/chunks_metadata.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"✅ Built chunked index: {len(meta)} chunks from {len(docs)} articles.")
