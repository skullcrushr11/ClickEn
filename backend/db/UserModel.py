from datetime import datetime
from typing import Any, Dict
import json
from bson import ObjectId
from pymongo.collection import Collection

class UserModel:
    def __init__(self, DB: Collection):
        self.db = DB
        self.collection = DB["users"]

    def create(self, user: Dict[str, Any]) -> str:
        """Creates a new user and returns JSON response."""
        user["createdAt"] = datetime.now()
        user["updatedAt"] = datetime.now()
        
        try:
            result = self.collection.insert_one(user)
            user["_id"] = str(result.inserted_id)
            user["createdAt"] = user["createdAt"].isoformat()
            user["updatedAt"] = user["updatedAt"].isoformat()
            return json.dumps({"message": "User created successfully", "status": 201, "data": user})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def get_all(self) -> str:
        """
        Fetches all users.
        """
        try:
            users = list(self.collection.find())
            for u in users:
                u["_id"] = str(u["_id"])  
                u["createdAt"] = user["createdAt"].isoformat()
                u["updatedAt"] = user["updatedAt"].isoformat()
            return json.dumps({"message": "Fetched all users", "status": 200, "data": users})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
    
    def get_by_id(self, user_id: str) -> str:
        """
        Fetch
        """
        try:
            user = self.collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return json.dumps({"error": "User not found", "status": 404})
            user["_id"] = str(user["_id"])
            user["createdAt"] = user["createdAt"].isoformat()
            user["updatedAt"] = user["updatedAt"].isoformat()
            return json.dumps({"message": "User fetched successfully", "status": 200, "data": user})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
    def update(self, user_id: str, user: Dict[str, Any]) -> str:
        """
        Updates an existing user.
        """
        user["updatedAt"] = datetime.now()
        
        try:
            updated = self.collection.update_one({"_id": ObjectId(user_id)}, {"$set": user})
            if updated.matched_count == 0:
                return json.dumps({"error": "User not found", "status": 404})
            
            user["createdAt"] = user["createdAt"].isoformat()
            user["updatedAt"] = user["updatedAt"].isoformat()
            return json.dumps({"message": "User fetched successfully", "status": 200, "data": self.get_by_id(user_id)})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
        
    def delete(self, user_id: str) -> str:
        """
        Deletes a user by ID.
        """
        try:
            user = self.get_by_id(user_id)
            if json.loads(user).get("status") == 404:
                return user  
            
            self.collection.delete_one({"_id": ObjectId(user_id)})
            return json.dumps({"message": "User deleted successfully", "status": 200})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})