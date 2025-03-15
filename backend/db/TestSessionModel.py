from datetime import datetime
from typing import Dict, Any
import json
from bson import ObjectId
from pymongo.collection import Collection

class TestSessionModel:
    def __init__(self, DB: Collection):
        self.db = DB
        self.collection = DB["test_sessions"]

    def create(self, test_session: Dict[str, Any]) -> str:
        """Creates a new test session and returns JSON response."""
        test_session["createdAt"] = datetime.now()
        test_session["updatedAt"] = datetime.now()
        
        try:
            result = self.collection.insert_one(test_session)
            test_session["_id"] = str(result.inserted_id)
            return json.dumps({"message": "Test session created successfully", "status": 201, "data": test_session})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def get_all(self) -> str:
        """Fetches all test sessions."""
        try:
            test_sessions = list(self.collection.find())
            for ts in test_sessions:
                ts["_id"] = str(ts["_id"])  # Convert ObjectId to string
            return json.dumps({"message": "Fetched all test sessions", "status": 200, "data": test_sessions})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
    
    def get_by_id(self, test_session_id: str) -> str:
        """Fetches a single test session by ID."""
        try:
            test_session = self.collection.find_one({"_id": ObjectId(test_session_id)})
            if not test_session:
                return json.dumps({"error": "Test session not found", "status": 404})
            test_session["_id"] = str(test_session["_id"])
            return json.dumps({"message": "Test session fetched successfully", "status": 200, "data": test_session})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
    def update(self, test_session_id: str, test_session: Dict[str, Any]) -> str:
        """Updates an existing test session."""
        test_session["updatedAt"] = datetime.now()
        
        try:
            updated = self.collection.update_one({"_id": ObjectId(test_session_id)}, {"$set": test_session})
            if updated.matched_count == 0:
                return json.dumps({"error": "Test session not found", "status": 404})
            
            return self.get_by_id(test_session_id)  # Return the updated test session
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
    def delete(self, test_session_id: str) -> str:
        """Deletes a test session by ID."""
        try:
            test_session = self.get_by_id(test_session_id)
            if json.loads(test_session).get("status") == 404:
                return test_session  # If not found, return the error JSON
            
            self.collection.delete_one({"_id": ObjectId(test_session_id)})
            return json.dumps({"message": "Test session deleted successfully", "status": 200})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
