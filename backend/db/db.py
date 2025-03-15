import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

def connect_to_mongo():
    """Connects to MongoDB and returns a JSON response."""
    db = None

    load_dotenv()
    MONGO_URI = os.getenv("VITE_MONGODB_URI")

    if not MONGO_URI:
        return None, json.dumps({"error": "MONGO_URI not found", "status": 500})

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)
        db = client["procturingsystems"]
        return db, json.dumps({"message": "Connected to MongoDB", "status": 200})
    except Exception as e:
        return None, json.dumps({"error": str(e), "status": 500})
    
if __name__ == "__main__":
    response = connect_to_mongo()
    print(response)  
