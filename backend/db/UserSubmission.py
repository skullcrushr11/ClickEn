from datetime import datetime
from typing import Dict, Any
import json
from bson import ObjectId
from pymongo.collection import Collection

class UserSubmissionModel:
    def __init__(self, DB: Collection):
        self.db = DB
        self.collection = DB["user_submissions"]

    def create(self, user_submission: Dict[str, Any]) -> str:
        """Creates a new user submission and returns JSON response."""
        user_submission["createdAt"] = datetime.now()
        user_submission["updatedAt"] = datetime.now()
        
        try:
            result = self.collection.insert_one(user_submission)
            user_submission["_id"] = str(result.inserted_id)
            return json.dumps({"message": "User submission created successfully", "status": 201, "data": user_submission})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def get_all(self) -> str:
        """Fetches all user submissions."""
        try:
            user_submissions = list(self.collection.find())
            for us in user_submissions:
                us["_id"] = str(us["_id"])  
            return json.dumps({"message": "Fetched all user submissions", "status": 200, "data": user_submissions})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
    
    def get_by_id(self, user_submission_id: str) -> str:
        """Fetches a single user submission by ID."""
        try:
            user_submission = self.collection.find_one({"_id": ObjectId(user_submission_id)})
            if not user_submission:
                return json.dumps({"error": "User submission not found", "status": 404})
            user_submission["_id"] = str(user_submission["_id"])
            return json.dumps({"message": "User submission fetched successfully", "status": 200, "data": user_submission})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
    def update(self, user_submission_id: str, user_submission: Dict[str, Any]) -> str:
        """Updates an existing user submission."""
        user_submission["updatedAt"] = datetime.now()
        
        try:
            updated = self.collection.update_one({"_id": ObjectId(user_submission_id)}, {"$set": user_submission})
            if updated.matched_count == 0:
                return json.dumps({"error": "User submission not found", "status": 404})
            
            return self.get_by_id(user_submission_id)  
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
    
    def delete(self, user_submission_id: str) -> str:
        """Deletes a user submission by ID."""
        try:
            deleted = self.collection.delete_one({"_id": ObjectId(user_submission_id)})
            if deleted.deleted_count == 0:
                return json.dumps({"error": "User submission not found", "status": 404})
            
            return json.dumps({"message": "User submission deleted successfully", "status": 200})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})