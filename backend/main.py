from flask import Flask, jsonify
from db import DB, User
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

with app.app_context():
    if DB is not None:
        print("Connected to MongoDB")
    else:
        print("Failed to connect to MongoDB")
        exit(1)
    

@app.route('/')
def hello_world():
    response = {"message": "Hello, Flask!", "status": "success"}
    return jsonify(response) 

#add a test route tp insert user data
@app.route('/users')
def get_users():
    users = [
        {"name": "John Doe", "email": "johndoe@gmail.com", "password": "password"},
    ]
    for user in users:
        response = User.create(user)
        print(response)  

    return jsonify({"message": "Users inserted successfully", "status": 201})
    

if __name__ == "__main__":
    app.run(debug=True)
