import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get the MongoDB URI, DB, and collection names from .env
uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

# Check for missing environment variables
if not uri or not db_name or not collection_name:
    raise ValueError("Missing one or more required environment variables: MONGODB_URI, DB_NAME, COLLECTION_NAME")

# Connect to MongoDB
client = MongoClient(uri)
db = client[db_name]
collection = db[collection_name]

# Insert a test document
test_doc = {"title": "MongoDB Test", "content": "Connection successful!"}
collection.insert_one(test_doc)

# Read back documents
print("Documents in collection:")
for doc in collection.find({}, {"_id": 0}):
    print(doc)
