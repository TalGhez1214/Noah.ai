import os
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()

# MongoDB setup
client_mongo = MongoClient(os.getenv("MONGODB_URI"))
db = client_mongo[os.getenv("DB_NAME")]
col = db[os.getenv("COLLECTION_NAME")]

# OpenAI setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Fetch articles from MongoDB
docs = list(
    col.find(
        {},  # no filter: get all docs
        {
            "_id": 0,       # exclude Mongo's default ID
            "title": 1,     # include title
            "url": 1,       # include URL
            "content": 1,   # include full text
            "author": 1,     # include author
            "published_at": 1  # include published date
        }
    )
)

# 2. Generate embeddings for each article
def embed_text(text):
    """
    This function generates an embedding for the given text using OpenAI's API.

    Args:
        text (str): The text to generate an embedding for.

    Returns:
        np.ndarray: The generated embedding as a NumPy array with size (1536,).
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# 3. Store embeddings + metadata
embeddings = [] # List to hold embeddings
meta = [] # List to hold metadata - articles content 
for doc in docs:
    if not doc.get("content"):
        continue
    content_vector = embed_text(doc["content"])
    embeddings.append(content_vector)
    meta.append(doc)

embeddings_np = np.vstack(embeddings)

# 4. Create FAISS index
dimension = embeddings_np.shape[1]  # embedding size
index = faiss.IndexFlatL2(dimension) # Create a flat index for L2 distance
index.add(embeddings_np) # Add embeddings to the index

# Save index & metadata
faiss.write_index(index, "./rag/data_indexing/indexes_and_metadata_files/articles.index") # Save the FAISS index to a file
# Save metadata to a pickle file for later retrieval
import pickle
with open("./rag/data_indexing/indexes_and_metadata_files/articles_metadata.pkl", "wb") as f:
    pickle.dump(meta, f) # Save metadata to a binary file

print(f"Stored {len(meta)} articles in FAISS index.")


