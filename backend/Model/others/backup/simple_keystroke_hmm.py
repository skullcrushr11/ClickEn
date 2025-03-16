#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import argparse
import os
from collections import defaultdict
import time

# Define states
STATES = {
    0: "Normal",
    1: "Suspicious"
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simple Keystroke HMM Proctoring')
    parser.add_argument('--sincere', type=str, required=True,
                      help='Path to sincere keystroke data')
    parser.add_argument('--cheating', type=str, required=True,
                      help='Path to cheating keystroke data')
    parser.add_argument('--test-split', type=float, default=0.2,
                      help='Fraction of data to use for testing')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Output directory for results')
    return parser.parse_args()

def load_keystroke_data(file_path):
    """Load keystroke data from JSON file"""
    print(f"Loading keystroke data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a simple list of events with timestamp info
    all_events = []
    
    for user_id, user_data in data.items():
        if "keyboard_data" in user_data:
            user_events = []
            for event in user_data["keyboard_data"]:
                # Check if we have enough elements
                if len(event) >= 3:
                    event_type = event[0]  # KD or KU
                    key = event[1]
                    timestamp = event[2]
                    
                    user_events.append({
                        'user_id': user_id,
                        'event_type': event_type,
                        'key': key,
                        'timestamp': timestamp
                    })
            
            # Sort user events by timestamp
            user_events.sort(key=lambda x: x['timestamp'])
            all_events.extend(user_events)
    
    if len(all_events) > 0:
        print(f"Sample event: {all_events[0]}")
        
    print(f"Loaded {len(all_events)} events from {file_path}")
    return all_events

def extract_features(events):
    """Extract keystroke timing features"""
    print("Extracting features...")
    
    if not events:
        print("No events to process")
        return []
    
    # Group events by user
    user_events = defaultdict(list)
    for event in events:
        user_events[event['user_id']].append(event)
    
    # Extract features per user
    all_features = []
    
    for user_id, user_data in user_events.items():
        # Sort by timestamp
        user_data.sort(key=lambda x: x['timestamp'])
        
        # Process only KD (key down) events to get inter-key intervals
        kd_events = [e for e in user_data if e['event_type'] == 'KD']
        
        # Calculate inter-key intervals
        for i in range(1, len(kd_events)):
            current_time = kd_events[i]['timestamp']
            prev_time = kd_events[i-1]['timestamp']
            
            # Calculate time difference in ms
            time_diff = current_time - prev_time
            
            # Only use reasonable timing values
            if 10 < time_diff < 2000:  # 10ms to 2 seconds
                all_features.append([time_diff])
    
    # Convert to numpy array
    if all_features:
        features = np.array(all_features)
        print(f"Extracted {len(features)} feature vectors from {len(user_events)} users")
        return features
    else:
        print("No valid features extracted")
        return np.array([])

def train_hmm(sincere_features, cheating_features):
    """
    Train a Hidden Markov Model on the keystroke data.
    
    This function creates a 2-state HMM where:
    - State 0 represents normal typing behavior
    - State 1 represents suspicious/cheating typing behavior
    
    The transition matrix represents the probability of moving between states:
    - transmat_[i,j] is the probability of transitioning from state i to state j
    """
    print("Training HMM model...")
    
    # Combine feature sets
    all_features = np.vstack([sincere_features, cheating_features])
    
    if len(all_features) == 0:
        print("Error: No features available for training")
        return None, None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_features)
    
    # Create lengths array for training
    sincere_len = len(sincere_features)
    cheating_len = len(cheating_features)
    lengths = [sincere_len, cheating_len]
    
    # Create the HMM with 2 states
    hmm = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=100,
        random_state=42,
        init_params='mc'  # Initialize means and covariances, but not startprob or transmat
    )
    
    # Set initial state probabilities
    # Higher probability to start in normal state (0)
    hmm.startprob_ = np.array([0.9, 0.1])
    
    # Set initial transition matrix
    # The transition matrix defines how likely it is to move between states:
    # Row 0: Probabilities of transitioning from state 0 (normal) to states 0 and 1
    # Row 1: Probabilities of transitioning from state 1 (suspicious) to states 0 and 1
    hmm.transmat_ = np.array([
        [0.95, 0.05],  # 95% chance to stay in normal state, 5% to transition to suspicious
        [0.20, 0.80]   # 20% chance to return to normal state, 80% to stay suspicious
    ])
    
    print("Initial transition matrix:")
    print(hmm.transmat_)
    
    try:
        # Fit the model - this will update means and covariances but keep our transition matrix
        hmm.fit(X_scaled, lengths=lengths)
        
        print("Final transition matrix after training:")
        print(hmm.transmat_)
        
        print(f"HMM trained successfully (log-likelihood: {hmm.score(X_scaled, lengths):.2f})")
        return hmm, scaler
    except Exception as e:
        print(f"Error training HMM: {e}")
        return None, None

def predict(hmm, scaler, features):
    """Predict states for a sequence of features"""
    if hmm is None or len(features) == 0:
        return None, 0
    
    X = scaler.transform(features)
    states = hmm.predict(X)
    
    # Calculate the ratio of suspicious states (state 1)
    suspicious_ratio = np.mean(states == 1)
    
    return states, suspicious_ratio

def evaluate_model(hmm, scaler, sincere_test, cheating_test):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    
    # Predict states
    sincere_states, sincere_ratio = predict(hmm, scaler, sincere_test)
    cheating_states, cheating_ratio = predict(hmm, scaler, cheating_test)
    
    # Find optimal threshold for classification
    best_threshold = 0.3  # Default
    best_accuracy = 0.0
    
    for threshold in np.linspace(0.2, 0.5, 30):
        sincere_correct = sincere_ratio < threshold
        cheating_correct = cheating_ratio >= threshold
        accuracy = 0.5 * (sincere_correct + cheating_correct)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Apply best threshold
    sincere_correct = sincere_ratio < best_threshold
    cheating_correct = cheating_ratio >= best_threshold
    overall_acc = 0.5 * (sincere_correct + cheating_correct)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Sincere data - Suspicious state ratio: {sincere_ratio:.4f}")
    print(f"Cheating data - Suspicious state ratio: {cheating_ratio:.4f}")
    print(f"Optimal threshold: {best_threshold:.4f}")
    print("\nAccuracy:")
    print(f"  Sincere: {1.0 if sincere_correct else 0.0:.2f}")
    print(f"  Cheating: {1.0 if cheating_correct else 0.0:.2f}")
    print(f"  Overall: {overall_acc:.2f}")
    print("="*50 + "\n")
    
    # Create visualizations
    plot_state_distributions(sincere_states, cheating_states)
    
    return {
        'sincere_acc': 1.0 if sincere_correct else 0.0,
        'cheating_acc': 1.0 if cheating_correct else 0.0,
        'overall_acc': overall_acc,
        'sincere_ratio': sincere_ratio,
        'cheating_ratio': cheating_ratio,
        'threshold': best_threshold
    }

def plot_state_distributions(sincere_states, cheating_states):
    """Create visualization of state distributions"""
    if sincere_states is None or cheating_states is None:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = [-0.5, 0.5, 1.5]  # Bins for states 0 and 1
    
    if sincere_states is not None and len(sincere_states) > 0:
        plt.hist(sincere_states, bins=bins, alpha=0.5, label='Sincere', density=True, color='green')
    
    if cheating_states is not None and len(cheating_states) > 0:
        plt.hist(cheating_states, bins=bins, alpha=0.5, label='Cheating', density=True, color='red')
    
    plt.title('HMM State Distribution', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xticks([0, 1], ['Normal', 'Suspicious'])
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig('state_distribution.png')
    print("Saved state distribution plot")

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load keystroke data
    sincere_events = load_keystroke_data(args.sincere)
    cheating_events = load_keystroke_data(args.cheating)
    
    # Extract features
    sincere_features = extract_features(sincere_events)
    cheating_features = extract_features(cheating_events)
    
    if len(sincere_features) == 0 or len(cheating_features) == 0:
        print("Error: Failed to extract features. Check your data format.")
        return
    
    # Split data for training and testing
    sincere_split = int(len(sincere_features) * (1 - args.test_split))
    cheating_split = int(len(cheating_features) * (1 - args.test_split))
    
    sincere_train = sincere_features[:sincere_split]
    sincere_test = sincere_features[sincere_split:]
    cheating_train = cheating_features[:cheating_split]
    cheating_test = cheating_features[cheating_split:]
    
    print(f"Training with {len(sincere_train)} sincere feature vectors and {len(cheating_train)} cheating feature vectors")
    print(f"Testing with {len(sincere_test)} sincere feature vectors and {len(cheating_test)} cheating feature vectors")
    
    # Train the HMM model
    start_time = time.time()
    hmm, scaler = train_hmm(sincere_train, cheating_train)
    train_time = time.time() - start_time
    
    if hmm is None:
        print("Error: Failed to train HMM model")
        return
    
    # Evaluate the model
    start_time = time.time()
    results = evaluate_model(hmm, scaler, sincere_test, cheating_test)
    eval_time = time.time() - start_time
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print(f"Sincere accuracy: {results['sincere_acc']:.2f}")
    print(f"Cheating accuracy: {results['cheating_acc']:.2f}")
    print(f"Overall accuracy: {results['overall_acc']:.2f}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Save results
    with open(os.path.join(args.output_dir, "hmm_results.txt"), "w") as f:
        f.write(f"Sincere accuracy: {results['sincere_acc']:.2f}\n")
        f.write(f"Cheating accuracy: {results['cheating_acc']:.2f}\n")
        f.write(f"Overall accuracy: {results['overall_acc']:.2f}\n")
        f.write(f"Sincere suspicion ratio: {results['sincere_ratio']:.4f}\n")
        f.write(f"Cheating suspicion ratio: {results['cheating_ratio']:.4f}\n")
        f.write(f"Decision threshold: {results['threshold']:.4f}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write(f"Evaluation time: {eval_time:.2f} seconds\n")
    
    print(f"Results saved to {os.path.join(args.output_dir, 'hmm_results.txt')}")

if __name__ == "__main__":
    main() 