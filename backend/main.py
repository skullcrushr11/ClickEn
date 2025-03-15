from flask import Flask, jsonify
from db import DB, User
from flask_cors import CORS
import json
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('keyevents')
def handle_keyevents(events):
    try:
        for event in events:
            print(f"Key event: {event['type']} {event['key']} at {event['timestamp']}")
        return {'status': 'ok', 'count': len(events)}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    socketio.run(app, debug=True)