from flask import Flask, jsonify, request
from db import DB, User
from flask_cors import CORS
import threading, time
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import os

global risk_score 
risk_score=0


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
event_buffer = []
buffer_lock = threading.Lock()
last_process_time = time.time()


  # Replace with your actual API key
COPYLEAKS_API_URL = "https://api.copyleaks.com/v2/writer-detector"

@app.route("/proxy/copyleaks", methods=["POST"])
def proxy_copyleaks():
    load_dotenv()
    VITE_MONGODB_URI = os.getenv("VITE_MONGODB_URI")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


    try:
        import requests
        import uuid  # For generating a unique scan ID

        # Replace with your actual Copyleaks access token
          

        # Copyleaks API endpoint
        API_URL = "https://api.copyleaks.com/v2/writer-detector"

        # Generate a unique scan ID for this request
        scan_id = str(uuid.uuid4())

        # Text prompt to analyze (Modify this as needed)
        text_prompt = """Artificial Intelligence (AI) has rapidly evolved in recent years, transforming multiple industries. 
        From healthcare to finance, AI's ability to analyze data and automate tasks has made it an essential tool in modern society. 
        For instance, in medicine, AI assists doctors in diagnosing diseases and predicting patient outcomes. AI-powered chatbots 
        enhance customer service by providing instant responses. Meanwhile, self-driving cars utilize AI to navigate roads safely. 

        Despite these benefits, AI raises concerns, particularly regarding ethics and job displacement. Many fear that as AI 
        systems become more advanced, human jobs will be at risk. Additionally, AI models sometimes exhibit biases, leading 
        to unfair treatment in decision-making processes such as hiring and lending. Ensuring that AI is transparent and 
        fair is crucial for responsible implementation.

        Furthermore, AI-generated content is now a major concern. As AI models like ChatGPT become more sophisticated, 
        it is increasingly difficult to differentiate between human-written and AI-generated text. Tools like Copyleaks 
        AI detection help verify authenticity, ensuring that content remains credible. Academic institutions, publishers, 
        and businesses now rely on such AI detection tools to prevent misinformation and maintain trust.

        In conclusion, AI presents both opportunities and challenges. While it enhances productivity and efficiency, 
        it also requires careful regulation and ethical considerations. Moving forward, it is vital to balance AI innovation 
        with responsible oversight to ensure its positive impact on society."""

        # API request headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"  # Use your token here
        }

        # API request payload
        payload = {
            "text": text_prompt,
            "sandbox": False,  # Set to True if using a testing environment
            "explain": True,  # Enables detailed analysis
            "sensitivity": 2  # Adjust detection sensitivity (0-5, default is 2)
        }

        # Send request to Copyleaks API
        response = requests.post(f"{API_URL}/{scan_id}/check", headers=headers, json=payload)

        # Print the API response
        if response.status_code == 200:
            result = response.json()
            print("\n✅ AI Detection Response:")
            print(result)
            
            # Extracting key details
            print("\n🔹 Model Version:", result.get("modelVersion"))
            print("🔹 AI Probability:", result.get("summary", {}).get("ai", "N/A"))
            print("🔹 Human Probability:", result.get("summary", {}).get("human", "N/A"))
            
            # Print detailed classification results
            for i, res in enumerate(result.get("results", [])):
                # print(f"\n🔍 Classification {i+1}:")
                # print("  - AI Classification:", res.get("classification"))
                print("  - Probability:", res.get("probability"))
            return 

        else:
            print("\n❌ Error:", response.status_code, response.text)


    except Exception as e:
        return jsonify({"error": "Error connecting to Copyleaks API", "details": str(e)}), 500

def process_events_batch():
    """
    Process all events collected in the last 10 seconds as a time series
    """
    global event_buffer
    global risk_score  # Add this line to properly reference the global variable
    
    with buffer_lock:
        # Make a copy of the buffer and clear it
        events_to_process = event_buffer.copy()
        event_buffer = []  # Clear the buffer
    
    # Only process if there are events
        
    if events_to_process:
        disallowed_hotkeys = [
            {'Control', 'shift', 'n'},  # New incognito window
            {'Control', 't'},  # New tab
            {'Control', 'tab'},  # Switch tab
            {'alt', 'tab'},  # Switch application (Windows)
            {'cmd', 'tab'},  # Switch application (Mac)
            {'Control', 'shift'},  # Task Manager (Windows)
            {'cmd', 'option', 'esc'},  # Force quit (Mac)
            {'Control', 'w'},  # Close tab
            {'alt', 'f4'},  # Close window (Windows)
            {'cmd', 'q'},  # Quit application (Mac)
            {'Control', 'p'},  # Print (potential for copying)
            {'Control', 's'},  # Save page
            {'Control', 'u'},  # View page source
            {'Control', 'f'},  # Find on page
            {'Control', 'h'},  # Open history
                ]
        
        print(f"\n>>> Processing batch of {len(events_to_process)} events collected in the last 10 seconds")
        
        def detect_violations(events):
            pressed_keys = set()
            violations = 0

            for event in events:
                key = event['key'].lower()
                
                if event['type'] == 'KD':
                    pressed_keys.add(key)
                
                if event['type'] == 'KU' and key in pressed_keys:
                    pressed_keys.remove(key)

                # Check if a forbidden hotkey combination was pressed
                for hotkey in disallowed_hotkeys:
                    if hotkey.issubset(pressed_keys):
                        violations += 1
                        print(f">>> Violation detected: {' + '.join(hotkey)}")
                        pressed_keys.clear()  # Reset to prevent counting the same shortcut multiple times

            return violations

        # Detect violations in the processed events
        violations_count = detect_violations(events_to_process)

        # Maintain risk score (properly using the global variable now)
        risk_score += 5*violations_count

        print(f">>> Total violations detected: {violations_count}")
        print(f">>> Updated risk score: {risk_score}")
        
        # Sort events by timestamp
        events_to_process.sort(key=lambda x: x['timestamp'])
        
        # Format as time series data
        time_series_data = []
        for event in events_to_process:
            event_type = event['type']
            
            # Convert timestamp from milliseconds to seconds if needed
            timestamp_in_seconds = event['timestamp'] / 1000.0 if event['timestamp'] > 946684800000 else event['timestamp']
            
            try:
                formatted_time = datetime.fromtimestamp(timestamp_in_seconds).strftime('%H:%M:%S.%f')[:-3]
                time_series_data.append({
                    'time': formatted_time,
                    'raw_time': timestamp_in_seconds,
                    'event': f"{event_type} | {event['key']}"
                })
            except Exception as e:
                print(f"Error formatting timestamp for event {event}: {str(e)}")
        
        # Print time series data only if we have any valid entries
        if time_series_data:
            print(">>> Key Event Time Series:")
            print(">>> Time | Event")
            print(">>> ---------------")
            for entry in time_series_data:
                print(f">>> {entry['time']} | {entry['event']}")
            
            # Calculate intervals between key presses if there are enough events
            if len(time_series_data) > 1:
                print("\n>>> Time intervals between consecutive events:")
                for i in range(1, len(time_series_data)):
                    interval = time_series_data[i]['raw_time'] - time_series_data[i-1]['raw_time']
                    print(f">>> {time_series_data[i-1]['event']} → {time_series_data[i]['event']}: {interval:.3f}s")
        
        print(f">>> Processing complete\n")
    else:
        print("\n>>> No events to process in the last 10 seconds\n")

# Background task to process events every 10 seconds
def background_processor():
    global last_process_time
    
    print("Background processor started - will process events every 10 seconds")
    
    while True:
        current_time = time.time()
        time_elapsed = current_time - last_process_time
        
        # If 10 seconds have passed, process the events
        if time_elapsed >= 10:
            process_events_batch()
            last_process_time = current_time
        
        # Sleep for a short time to avoid high CPU usage
        time.sleep(0.5)

# Model and scaler for keystroke biometrics
model = None
scaler = None

# Dictionary to store keystroke data for each client session
client_keystroke_data = {}

def load_model():
    global model, scaler
    try:
        # Load the model
        model = tf.keras.models.load_model("HMM/keystroke_cheating_detector.h5")
        
        # Create scaler from training data
        scaler = recreate_scaler_from_training_data()
        
        print("Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def recreate_scaler_from_training_data():
    try:
        # Load training data
        normal_data = load_json("HMM/normal.json")
        cheating_data = load_json("HMM/cheating.json")
        df_normal = process_json(normal_data, label=0)
        df_cheating = process_json(cheating_data, label=1)
        df = pd.concat([df_normal, df_cheating], ignore_index=True)
        df.dropna(inplace=True)

        features = ["key_hold_time", "inter_key_delay"]

        scaler = MinMaxScaler()
        scaler.fit(df[features])
        return scaler
    except Exception as e:
        print(f"Error recreating scaler: {e}")
        return None

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

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
        inter_key_delay = key_down_time - kd_events[idx - 1][2] if idx > 0 else 0
        data.append(
            {
                "key": key,
                "key_hold_time": hold_time,
                "inter_key_delay": inter_key_delay,
                "label": label,
            }
        )
    return pd.DataFrame(data)

def process_json(data, label):
    df_list = []
    for session_id, session in data.items():
        df_session = extract_features_from_session(session, label)
        df_list.append(df_session)
    return pd.concat(df_list, ignore_index=True)

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
        # 0-1 scale
        "model_confidence": float(abs(cheating_probability - 0.5) * 2),
    }

def analyze_keystroke_session(session_data, threshold=0.5):
    """Analyze keystroke data and return risk assessment"""
    global model, scaler
    
    if model is None or scaler is None:
        return {"error": "Model not loaded"}, 500
    
    # Process keystroke data
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
        inter_key_delay = key_down_time - kd_events[idx - 1][2] if idx > 0 else 0
        data.append(
            {
                "key": key,
                "key_hold_time": hold_time,
                "inter_key_delay": inter_key_delay,
            }
        )

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    if df.empty:
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "risk_score": 0,
            "risk_level": "Unknown",
            "details": "Not enough keystroke data",
            "risk_factors": {}
        }

    # Prepare data for the model
    features = ["key_hold_time", "inter_key_delay"]
    df_normalized = df.copy()
    df_normalized[features] = scaler.transform(df[features])

    # Create sequences
    X_test = []
    seq_length = 20
    for i in range(len(df_normalized) - seq_length + 1):
        seq = df_normalized[features].values[i : i + seq_length]
        X_test.append(seq)

    X_test = np.array(X_test)

    if len(X_test) == 0:
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "risk_score": 0,
            "risk_level": "Unknown",
            "details": "Not enough keystroke sequences",
            "risk_factors": {}
        }

    # Get predictions
    predictions = model.predict(X_test, verbose=0)

    # Calculate standard metrics
    avg_prediction = np.mean(predictions)
    confidence = abs(avg_prediction - 0.5) * 2  # Scale to 0-1
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
    
with app.app_context():
    if DB is not None:
        print("Connected to MongoDB")
    else:
        print("Failed to connect to MongoDB")
        exit(1)
    
    # Load keystroke analysis model
    if not load_model():
        print("Warning: Could not load keystroke analysis model")

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

@app.route('/analyze_keystrokes', methods=['POST'])
def analyze_keystrokes():
    """
    Endpoint to analyze keystroke data and assess risk
    Expected JSON format:
    {
        "session_id": "unique_client_id",
        "keyboard_data": [
            ["KD", "a", 1620000000.123], 
            ["KU", "a", 1620000000.223],
            ...
        ],
        "reset_session": false  # Optional: reset the accumulated data
    }
    """
    if request.method == 'POST':
        try:
            data = request.json
            
            if not data or "keyboard_data" not in data:
                return jsonify({
                    "error": "Invalid data format",
                    "message": "Request must include 'keyboard_data' field"
                }), 400
            
            # Get or create session ID
            session_id = data.get("session_id", request.remote_addr)
            
            # Check if we need to reset the session
            if data.get("reset_session", False):
                if session_id in client_keystroke_data:
                    del client_keystroke_data[session_id]
            
            # Add new events to the session
            if session_id not in client_keystroke_data:
                client_keystroke_data[session_id] = []
            
            client_keystroke_data[session_id].extend(data["keyboard_data"])
            
            # Create session data with accumulated events
            session_data = {"keyboard_data": client_keystroke_data[session_id].copy()}
            
            # Analyze the keystroke data with accumulated history
            result = analyze_keystroke_session(session_data)
            
            # Add metadata about the session
            result["total_events"] = len(client_keystroke_data[session_id])
            result["session_id"] = session_id
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                "error": "Server error",
                "message": str(e)
            }), 500
    
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Initialize empty keystroke data for new client
    client_id = request.sid
    client_keystroke_data[client_id] = []
    print(f'New client connected: {client_id}')

@socketio.on('disconnect')
def handle_disconnect():
    # Clean up keystroke data for disconnected client
    client_id = request.sid
    if client_id in client_keystroke_data:
        del client_keystroke_data[client_id]
    print(f'Client disconnected: {client_id}')

@socketio.on('keyevents')
def handle_keyevents(events):
    global event_buffer
    try:
        # Keep the original logging functionality
        for event in events:
            print(event)
            # Ensure timestamp exists
            if 'timestamp' not in event:
                event['timestamp'] = time.time()
                
            # Log the event in the format KD/KU | key
            event_type = event['type']
            print(f"Key event: {event_type} | {event['key']} at {event['timestamp']}")
            
            # Add to the buffer for batch processing
            with buffer_lock:
                event_buffer.append(event)
                
        return {'status': 'ok', 'count': len(events)}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'status': 'error', 'message': str(e)}

@socketio.on('reset_session')
def handle_reset_session():
    """Reset the keystroke history for a client"""
    client_id = request.sid
    if client_id in client_keystroke_data:
        client_keystroke_data[client_id] = []
    return {'status': 'ok', 'message': 'Session reset successfully'}

if __name__ == "__main__":
    # Start the background processor thread before running the app
    processor_thread = threading.Thread(target=background_processor, daemon=True)
    processor_thread.start()
    
    socketio.run(app, debug=True)