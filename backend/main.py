from flask import Flask, jsonify
from db.db import connect_to_mongo
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

with app.app_context():
    str_res = connect_to_mongo()
    res = json.loads(str_res)
    
    if res["status"]  == 500:
        print(res["error"])
        exit(1)
    print("Connected to MongoDB")
    

@app.route('/')
def hello_world():
    response = {"message": "Hello, Flask!", "status": "success"}
    return jsonify(response) 

if __name__ == "__main__":
    app.run(debug=True)
