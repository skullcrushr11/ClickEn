import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

DB = None  

def connect_to_mongo():
    """Connects to MongoDB and returns a JSON response."""
    global DB  

    load_dotenv()
    MONGO_URI = os.getenv("VITE_MONGODB_URI")

    if not MONGO_URI:
        return json.dumps({"error": "MONGO_URI not found", "status": 500})

    try:
        client = MongoClient(MONGO_URI)
        DB = client["procturingsystems"]
        return json.dumps({"message": "Connected to MongoDB", "status": 200})
    except Exception as e:
        return json.dump({"error": str(e), "status": 500})

def get_collection(collection_name):
    """Returns a collection from the connected database."""
    if DB is None:
        return {"error": "Database is not connected", "status": 500}

    return json.dumps({"message": "Fetched the collection succesfully", "status": 200, collections: DB[collection_name]})

if __name__ == "__main__":
    response = connect_to_mongo()
    print(response)  
