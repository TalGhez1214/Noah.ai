from pymongo import MongoClient
import os
from datetime import timezone

MONGO_URI     = os.getenv("MONGODB_URI", "mongodb+srv://...")
DB_NAME       = os.getenv("DB_NAME", "content_db")
CHUNKS_COL    = os.getenv("CHUNKS_COL", "article_chunks")
ARTICLES_COL  = os.getenv("ARTICLES_COL", "articles")

class MongoDBClient():
    def __init__(self):
        self._client = MongoClient(MONGO_URI, tz_aware=True, tzinfo=timezone.utc)
        self._db = self._client[DB_NAME]
        self._chunks = self._db[CHUNKS_COL]
        self._articles = self._db[ARTICLES_COL]

MongoClientInstance = MongoDBClient()