import os, time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, ASCENDING, TEXT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Read .env file and make variables available via os.getenv

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "news")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "jpost_articles")
RSS_URL = os.getenv("RSS_URL", "https://www.jpost.com/rss/rssfeedsfrontpage.aspx")

# ==============================
# CONNECT TO MONGODB
# ==============================

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# Create indexes for:
# - Unique URL (no duplicate articles)
# - Text search across title/content
col.create_index([("url", ASCENDING)], unique=True)
col.create_index([("title", TEXT), ("content", TEXT)], name="text_search")

# HTTP request header to identify our script
HEADERS = {
    "User-Agent": "Mozilla/5.0 (+news-loader; Contact: you@example.com)"
}