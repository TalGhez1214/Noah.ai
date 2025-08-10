import os
from openai import OpenAI
import numpy as np

# OpenAI setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to create embedding for the query
def embed_query(query):
    """
    This function generates an embedding for the given query using OpenAI's API.

    Args:   
        query (str): The query to generate an embedding for.

    Returns:
        np.ndarray: The generated embedding as a NumPy array with size (1536,).
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)