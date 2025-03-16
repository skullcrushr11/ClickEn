import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Load JSON data


def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


normal_data = load_json("normal.json")
cheating_data = load_json("cheating.json")

# Extract features function remains the same


def extract_features_from_session(session, label):
    events = session.get("keyboard_data", [])
    data = []
    kd_events = []
    # Collect all key down events with their index and timestamp
    for i, event in enumerate(events):
        if event[0] == "KD":
            kd_events.append((i, event[1], event[2]))

    # Process each key down event
    for idx, (i, key, key_down_time) in enumerate(kd_events):
        key_up_time = None
        # Look for the first key up event for the same key after this key down
        for event in events[i:]:
            if event[0] == "KU" and event[1] == key:
                key_up_time = event[2]
                break
        # Compute key hold time if key up was found
        hold_time = key_up_time - key_down_time if key_up_time is not None else np.nan

        # Compute inter-key delay
        inter_key_delay = key_down_time - \
            kd_events[idx - 1][2] if idx > 0 else 0

        data.append({
            "key": key,
            "key_hold_time": hold_time,
            "inter_key_delay": inter_key_delay,
            # Add more features that might help
            # Convert key to ASCII
            "key_code": ord(key) if len(key) == 1 else 0,
            "label": label,
        })
    return pd.DataFrame(data)

# Process all sessions


def process_json(data, label):
    df_list = []
    for session_id, session in data.items():
        df_session = extract_features_from_session(session, label)
        if not df_session.empty:
            # Track session for better sequence creation
            df_session['session_id'] = session_id
            df_list.append(df_session)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# Process data
df_normal = process_json(normal_data, label=0)
df_cheating = process_json(cheating_data, label=1)

# Print class distribution
print(f"Normal samples: {len(df_normal)}")
print(f"Cheating samples: {len(df_cheating)}")

# Combine datasets and drop any rows with missing hold times
df = pd.concat([df_normal, df_cheating], ignore_index=True)
df.dropna(inplace=True)

# Feature engineering - add more features
df['key_hold_to_delay_ratio'] = df['key_hold_time'] / \
    df['inter_key_delay'].replace(0, 0.001)

# Normalize numerical features
features = ["key_hold_time", "inter_key_delay",
            "key_code", "key_hold_to_delay_ratio"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Create sequences by session to maintain temporal integrity


def create_sequences_by_session(df, seq_length=20):
    X, y = [], []

    for session_id in df['session_id'].unique():
        session_data = df[df['session_id'] == session_id]
        session_features = session_data[features].values
        session_labels = session_data['label'].values

        # Create sequences for this session
        for i in range(len(session_features) - seq_length):
            X.append(session_features[i:i + seq_length])
            # Use label of last item in sequence
            y.append(session_labels[i + seq_length - 1])

    return np.array(X), np.array(y)


# Create sequences with larger window
X, y = create_sequences_by_session(df, seq_length=20)

# Split data properly into train/validation/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Improved LSTM Model with regularization
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
         recurrent_dropout=0.2),
    BatchNormalization(),
    LSTM(64, return_sequences=True, recurrent_dropout=0.2),
    BatchNormalization(),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Use a more appropriate optimizer with a good learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(
    ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add callbacks to improve training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=0.00001, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1)
]

# Train with class weights and callbacks
history = model.fit(
    X_train, y_train,
    epochs=30,  # More epochs with early stopping
    batch_size=64,  # Try a larger batch size
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate on test set
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_results[0]}, Test accuracy: {test_results[1]}")

# Save the trained model
model.save("improved_keystroke_cheating_detector.h5")

# Plot training history to visualize improvement

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
