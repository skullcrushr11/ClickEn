import json
from datetime import datetime
from typing import Dict, Any
from bson import ObjectId
from db import DB

class QuestionModel:
    def __init__(self):
        self.db = DB
        self.collection = DB["questions"]

    def create(self, question: Dict[str, Any]) -> str:
        """Creates a new question and returns JSON response."""
        question["createdAt"] = datetime.now()
        question["updatedAt"] = datetime.now()
        
        try:
            result = self.collection.insert_one(question)
            question["_id"] = str(result.inserted_id)
            return json.dumps({"message": "Question created successfully", "status": 201, "data": question})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def get_all(self) -> str:
        """Fetches all questions."""
        try:
            questions = list(self.collection.find())
            for q in questions:
                q["_id"] = str(q["_id"])  # Convert ObjectId to string
            return json.dumps({"message": "Fetched all questions", "status": 200, "data": questions})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def get_by_id(self, question_id: str) -> str:
        """Fetches a single question by ID."""
        try:
            question = self.collection.find_one({"_id": ObjectId(question_id)})
            if not question:
                return json.dumps({"error": "Question not found", "status": 404})
            question["_id"] = str(question["_id"])
            return json.dumps({"message": "Question fetched successfully", "status": 200, "data": question})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def update(self, question_id: str, question: Dict[str, Any]) -> str:
        """Updates an existing question."""
        question["updatedAt"] = datetime.now()
        
        try:
            updated = self.collection.update_one({"_id": ObjectId(question_id)}, {"$set": question})
            if updated.matched_count == 0:
                return json.dumps({"error": "Question not found", "status": 404})
            
            return self.get_by_id(question_id)  # Return the updated question
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})

    def delete(self, question_id: str) -> str:
        """Deletes a question by ID."""
        try:
            question = self.get_by_id(question_id)
            if json.loads(question).get("status") == 404:
                return question  # If not found, return the error JSON
            
            self.collection.delete_one({"_id": ObjectId(question_id)})
            return json.dumps({"message": "Question deleted successfully", "status": 200})
        except Exception as e:
            return json.dumps({"error": str(e), "status": 500})
