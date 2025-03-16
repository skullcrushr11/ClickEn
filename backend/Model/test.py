import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import os

# -------------------------------------------
# Helper functions for JSON processing
# -------------------------------------------


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


# -------------------------------------------
# Functions for preparing test data and model evaluation
# -------------------------------------------


def prepare_test_data(
    test_session,
    scaler,
    features=["key_hold_time", "inter_key_delay"],
    seq_length=20,
):
    events = test_session.get("keyboard_data", [])
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
    # Derived feature
    df["key_hold_to_delay_ratio"] = df["key_hold_time"] / df["inter_key_delay"].replace(
        0, 0.001
    )
    # Normalize using the provided scaler
    df[features] = scaler.transform(df[features])
    # Create sequences for model input
    sequences = []
    for i in range(len(df) - seq_length + 1):
        seq = df[features].values[i : i + seq_length]
        sequences.append(seq)
    return np.array(sequences) if sequences else np.array([])


def calculate_risk_score(predictions, keystroke_data, scaling_factor=100):
    """
    Calculate a dynamic risk score based on model predictions and keystroke patterns.

    Args:
        predictions: Array of prediction values from the model
        keystroke_data: DataFrame with processed keystroke data
        scaling_factor: Factor to scale the risk score (default: 100)

    Returns:
        risk_score: Integer between 0-100 indicating risk level
        risk_factors: Dictionary with specific risk indicators
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


def test_single_session(session_data, scaler, model, threshold=0.5):
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
        }

    # Get predictions
    predictions = model.predict(X_test)

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


# -------------------------------------------
# Main script
# -------------------------------------------
# Load the saved model
model = tf.keras.models.load_model("keystroke_cheating_detector.h5")

# Recreate the scaler from training data
scaler = recreate_scaler_from_training_data()

# Load the normal and cheating JSON files
normal_data = load_json("normal.json")
cheating_data = load_json("cheating.json")

# Create a test session by merging parts of one normal and one cheating session.
# Here, we pick the first session from each file.
normal_session = list(normal_data.values())[0]
cheating_session = list(cheating_data.values())[0]

# Optionally, take only a subset of events from each session.
merged_keyboard_data = normal_session.get("keyboard_data", []) + cheating_session.get(
    "keyboard_data", []
)

# Form a test session dictionary
test_session = {"keyboard_data": merged_keyboard_data}

# Save the test session to test_session.json to ensure the file exists for loading later.
test_session_filename = "test_session.json"
with open(test_session_filename, "w") as f:
    json.dump(test_session, f)

# Optionally, check that the file was saved successfully.
if not os.path.exists(test_session_filename):
    raise FileNotFoundError(f"{test_session_filename} not found.")

# Load the test session from file (as originally intended)
with open(test_session_filename, "r") as file:
    loaded_test_session = json.load(file)

# Test the model on the loaded test session
result = test_single_session(loaded_test_session, scaler, model)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Average Score: {result['average_score']:.2f}")
print(f"Details: {result['details']}")
