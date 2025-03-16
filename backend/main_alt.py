from flask import Flask, jsonify, request
from db import DB, User
from flask_cors import CORS
import json
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
import threading
import os

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for model and scaler
model = None
scaler = None
keystroke_sessions = {}  # Dictionary to store active sessions by user ID

# Initialize the model and scaler


def initialize_model():
    global model, scaler
    try:
        model = tf.keras.models.load_model("keystroke_cheating_detector.h5")
        scaler = recreate_scaler_from_training_data()
        print("Model and scaler initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")


def recreate_scaler_from_training_data():
    # Load training data from both normal and cheating JSON files
    normal_data = load_json("normal.json")
    cheating_data = load_json("cheating.json")
    df_normal = process_json(normal_data, label=0)
    df_cheating = process_json(cheating_data, label=1)
    df = pd.concat([df_normal, df_cheating], ignore_index=True)
    df.dropna(inplace=True)

    features = ["key_hold_time", "inter_key_delay"]

    scaler = MinMaxScaler()
    scaler.fit(df[features])
    return scaler


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def process_json(data, label):
    df_list = []
    for session_id, session in data.items():
        df_session = extract_features_from_session(session, label)
        df_list.append(df_session)
    return pd.concat(df_list, ignore_index=True)


def extract_features_from_session(session, label):
    events = session.get("keyboard_data", [])
    data = []
    kd_events = []
    # Collect key down events with index and timestamp
    for i, event in enumerate(events):
        if event[0] == "KD":
            kd_events.append((i, event[1], event[2]))
    # Process each key down event to compute hold time and inter-key delay
    for idx, (i, key, key_down_time) in enumerate(kd_events):
        key_up_time = None
        for event in events[i:]:
            if event[0] == "KU" and event[1] == key:
                key_up_time = event[2]
                break
        hold_time = key_up_time - key_down_time if key_up_time is not None else np.nan
        inter_key_delay = key_down_time - \
            kd_events[idx - 1][2] if idx > 0 else 0
        data.append(
            {
                "key": key,
                "key_hold_time": hold_time,
                "inter_key_delay": inter_key_delay,
                "label": label,
            }
        )
    return pd.DataFrame(data)


def calculate_risk_score(predictions, keystroke_data, scaling_factor=100):
    """
    Calculate a dynamic risk score based on model predictions and keystroke patterns.
    """
    # Base score from model predictions (0-100)
    cheating_probability = float(np.mean(predictions))
    base_score = cheating_probability * scaling_factor

    # Additional risk factors analysis
    risk_factors = {}

    # 1. Consistency of typing patterns
    if len(keystroke_data) > 10:  # Only calculate if enough data
        hold_time_std = keystroke_data["key_hold_time"].std()
        delay_time_std = keystroke_data["inter_key_delay"].std()

        # High variation could indicate multiple typists or automated tools
        typing_consistency = max(0, min(30, (hold_time_std * 10)))
        risk_factors["typing_consistency"] = typing_consistency

        # 2. Sudden changes in typing speed
        typing_delays = keystroke_data["inter_key_delay"].values
        if len(typing_delays) > 20:
            # Look for large jumps in typing speed
            delay_diffs = np.abs(np.diff(typing_delays))
            unusual_pauses = np.sum(delay_diffs > np.mean(typing_delays) * 5)
            pause_factor = min(20, unusual_pauses * 2)
            risk_factors["unusual_pauses"] = pause_factor
        else:
            risk_factors["unusual_pauses"] = 0

        # 3. Analyze sequence of prediction values for suspicious patterns
        if len(predictions) > 1:
            prediction_changes = np.abs(np.diff(predictions.flatten()))
            # Threshold for significant change
            abrupt_changes = np.sum(prediction_changes > 0.4)
            change_factor = min(20, abrupt_changes * 4)
            risk_factors["behavior_changes"] = change_factor
        else:
            risk_factors["behavior_changes"] = 0
    else:
        # Not enough data
        risk_factors["typing_consistency"] = 0
        risk_factors["unusual_pauses"] = 0
        risk_factors["behavior_changes"] = 0

    # Calculate weighted risk score
    # Base: 70% from model, 30% from additional factors
    additional_risk = sum(risk_factors.values())
    weighted_risk = (base_score * 0.7) + (additional_risk * 0.3)

    # Cap between 0-100
    final_risk_score = max(0, min(100, weighted_risk))

    # Risk level categorization
    if final_risk_score < 20:
        risk_level = "Very Low"
    elif final_risk_score < 40:
        risk_level = "Low"
    elif final_risk_score < 60:
        risk_level = "Moderate"
    elif final_risk_score < 80:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "risk_score": int(final_risk_score),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "model_confidence": float(abs(cheating_probability - 0.5) * 2),
    }


def analyze_keystroke_session(session_data):
    """
    Analyze a keystroke session and return risk assessment
    """
    global model, scaler

    if model is None or scaler is None:
        return {"error": "Model not initialized"}

    events = session_data.get("keyboard_data", [])
    data = []
    kd_events = []

    # Collect key down events
    for i, event in enumerate(events):
        if event[0] == "KD":
            kd_events.append((i, event[1], event[2]))

    # Process each key down event
    for idx, (i, key, key_down_time) in enumerate(kd_events):
        key_up_time = None
        for event in events[i:]:
            if event[0] == "KU" and event[1] == key:
                key_up_time = event[2]
                break
        hold_time = key_up_time - key_down_time if key_up_time is not None else np.nan
        inter_key_delay = key_down_time - \
            kd_events[idx - 1][2] if idx > 0 else 0
        data.append({
            "key": key,
            "key_hold_time": hold_time,
            "inter_key_delay": inter_key_delay,
        })

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    if df.empty:
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "risk_score": 0,
            "risk_level": "Unknown",
            "details": "Not enough keystroke data",
        }

    # Prepare data for the model
    features = ["key_hold_time", "inter_key_delay"]
    df_normalized = df.copy()
    df_normalized[features] = scaler.transform(df[features])

    # Create sequences
    X_test = []
    seq_length = 20
    for i in range(len(df_normalized) - seq_length + 1):
        seq = df_normalized[features].values[i: i + seq_length]
        X_test.append(seq)

    X_test = np.array(X_test)

    if len(X_test) == 0:
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "risk_score": 0,
            "risk_level": "Unknown",
            "details": "Not enough keystroke sequences",
        }

    # Get predictions
    predictions = model.predict(X_test, verbose=0)

    # Calculate standard metrics
    avg_prediction = np.mean(predictions)
    confidence = abs(avg_prediction - 0.5) * 2  # Scale to 0-1
    threshold = 0.5
    classification = "Cheating" if avg_prediction > threshold else "Normal"

    # Calculate risk score
    risk_assessment = calculate_risk_score(predictions, df)

    # Count prediction types
    prediction_counts = {
        "normal_sequences": int(np.sum(predictions < threshold)),
        "cheating_sequences": int(np.sum(predictions >= threshold)),
        "total_sequences": len(predictions),
    }

    # Return comprehensive results
    return {
        "prediction": classification,
        "confidence": float(confidence),
        "average_score": float(avg_prediction),
        "risk_score": risk_assessment["risk_score"],
        "risk_level": risk_assessment["risk_level"],
        "risk_factors": risk_assessment["risk_factors"],
        "details": prediction_counts,
    }


# Initialize DB connection
with app.app_context():
    if DB is not None:
        print("Connected to MongoDB")
    else:
        print("Failed to connect to MongoDB")
        exit(1)
    # Initialize model
    initialize_model()


@app.route('/')
def hello_world():
    response = {"message": "Hello, Flask!", "status": "success"}
    return jsonify(response)


@app.route('/users')
def get_users():
    users = [
        {"name": "John Doe", "email": "johndoe@gmail.com", "password": "password"},
    ]
    for user in users:
        response = User.create(user)
        print(response)

    return jsonify({"message": "Users inserted successfully", "status": 201})

# New route to get risk assessment for a specific user session


@app.route('/risk-assessment/<user_id>', methods=['GET'])
def get_risk_assessment(user_id):
    if user_id not in keystroke_sessions:
        return jsonify({"error": "No keystroke data found for this user"}), 404

    # Analyze the current keystroke session
    assessment = analyze_keystroke_session(keystroke_sessions[user_id])
    return jsonify(assessment)

# New route to clear keystroke data for a specific user


@app.route('/clear-keystrokes/<user_id>', methods=['POST'])
def clear_keystrokes(user_id):
    if user_id in keystroke_sessions:
        keystroke_sessions[user_id] = {"keyboard_data": []}
        return jsonify({"message": "Keystroke data cleared successfully"})
    return jsonify({"error": "User not found"}), 404


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('keyevents')
def handle_keyevents(events):
    try:
        # Extract user ID from the first event
        user_id = events[0].get(
            'userId') if events and 'userId' in events[0] else 'anonymous'

        # Initialize session if it doesn't exist
        if user_id not in keystroke_sessions:
            keystroke_sessions[user_id] = {"keyboard_data": []}

        # Process each key event
        for event in events:
            event_type = event['type']
            key = event['key']
            timestamp = event['timestamp']

            # Add to session data
            if event_type == 'keydown':
                keystroke_sessions[user_id]["keyboard_data"].append(
                    ["KD", key, timestamp])
            elif event_type == 'keyup':
                keystroke_sessions[user_id]["keyboard_data"].append(
                    ["KU", key, timestamp])

            print(f"Key event: {event_type} {key} at {timestamp}")

        # If we have enough data, analyze the session
        if len(keystroke_sessions[user_id]["keyboard_data"]) >= 40:  # Minimum threshold
            assessment = analyze_keystroke_session(keystroke_sessions[user_id])

            # Emit the risk assessment back to the client
            socketio.emit('risk_assessment', {
                'userId': user_id,
                'assessment': assessment
            })

            # Log the assessment
            print(f"Risk assessment for {user_id}: {
                  assessment['risk_level']} ({assessment['risk_score']})")

        return {'status': 'ok', 'count': len(events)}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


if __name__ == "__main__":
    socketio.run(app, debug=True)
