import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import os
import time
import keyboard
import threading
import sys

# Load the model and prepare the scaler
model = tf.keras.models.load_model("keystroke_cheating_detector.h5")


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


def calculate_risk_score(predictions, keystroke_data, scaling_factor=100):
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


class KeystrokeMonitor:
    def __init__(self, scaler, model, seq_length=20, update_interval=5):
        self.scaler = scaler
        self.model = model
        self.seq_length = seq_length
        self.update_interval = update_interval  # seconds between risk score updates
        self.keyboard_data = []
        self.key_states = {}  # To track which keys are currently pressed
        self.running = False
        self.lock = threading.Lock()

    def on_key_press(self, event):
        key = event.name
        timestamp = time.time()
        with self.lock:
            # Only record if key wasn't already pressed
            if key not in self.key_states or not self.key_states[key]:
                self.keyboard_data.append(["KD", key, timestamp])
                self.key_states[key] = True

    def on_key_release(self, event):
        key = event.name
        timestamp = time.time()
        with self.lock:
            if key in self.key_states and self.key_states[key]:
                self.keyboard_data.append(["KU", key, timestamp])
                self.key_states[key] = False

    def start_monitoring(self):
        self.running = True
        keyboard.on_press(self.on_key_press)
        keyboard.on_release(self.on_key_release)
        update_thread = threading.Thread(target=self.periodic_risk_assessment)
        update_thread.daemon = True
        update_thread.start()

        print("\n===== REAL-TIME KEYSTROKE ANALYSIS =====")
        print("Start typing to see your risk assessment")
        print("Press Ctrl+C to exit\n")

        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_monitoring()
            print("\nKeyboard monitoring stopped.")

    def stop_monitoring(self):
        self.running = False
        keyboard.unhook_all()

    def periodic_risk_assessment(self):
        while self.running:
            time.sleep(self.update_interval)
            self.calculate_and_display_risk()

    def calculate_and_display_risk(self):
        with self.lock:
            current_data = self.keyboard_data.copy()

        if len(current_data) < 10:  # Need at least a few keystrokes
            return

        # Process the keystroke data
        session = {"keyboard_data": current_data}
        result = self.assess_session(session)

        # Clear the terminal and display the latest assessment
        os.system("cls" if os.name == "nt" else "clear")

        print("\n===== REAL-TIME KEYSTROKE ANALYSIS =====")
        print(f"Total keystrokes recorded: {len(current_data)}")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Model Confidence: {result['confidence']:.2f}")

        if "risk_factors" in result:
            print("\nRisk Factors:")
            for factor, value in result["risk_factors"].items():
                print(f"  - {factor.replace('_', ' ').title()}: {value:.2f}")

        print("\nContinue typing... (Press Ctrl+C to exit)")

    def assess_session(self, session_data):
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
            hold_time = (
                key_up_time - key_down_time if key_up_time is not None else np.nan
            )
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
        df_normalized[features] = self.scaler.transform(df[features])

        # Create sequences
        X_test = []
        for i in range(len(df_normalized) - self.seq_length + 1):
            seq = df_normalized[features].values[i : i + self.seq_length]
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
        predictions = self.model.predict(X_test, verbose=0)

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


def main():
    try:
        # Initialize the model and scaler
        print("Loading model and initializing scaler...")
        scaler = recreate_scaler_from_training_data()

        # Create keystroke monitor
        monitor = KeystrokeMonitor(scaler, model)

        # Start monitoring
        monitor.start_monitoring()

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
