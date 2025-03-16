#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simple Keystroke Pattern Comparison')
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
    
    # Create a list of events with user and timestamp info
    all_events = []
    
    for user_id, user_data in data.items():
        if "keyboard_data" in user_data:
            for event in user_data["keyboard_data"]:
                if len(event) >= 3:
                    all_events.append({
                        'user_id': user_id,
                        'event_type': event[0],  # KD or KU
                        'key': event[1],
                        'timestamp': event[2]
                    })
    
    # Sort all events by timestamp
    all_events.sort(key=lambda x: x['timestamp'])
    
    if len(all_events) > 0:
        print(f"Sample event: {all_events[0]}")
    
    print(f"Loaded {len(all_events)} events from {file_path}")
    return all_events

def create_time_windows(events, primary_window=45000, secondary_window=180000):
    """
    Create primary (45s) and secondary (3min) windows for comparison.
    
    Args:
        events: List of keystroke events
        primary_window: Duration of primary window in ms (default: 45s)
        secondary_window: Duration of secondary window in ms (default: 3min)
        
    Returns:
        Dictionary mapping user_ids to their primary and secondary window events
    """
    print(f"Creating {primary_window/1000}s primary and {secondary_window/1000}s secondary time windows")
    
    # Group events by user
    user_events = defaultdict(list)
    for event in events:
        user_events[event['user_id']].append(event)
    
    windows = {}
    
    for user_id, user_data in user_events.items():
        if len(user_data) < 20:  # Skip users with very few events
            continue
            
        # Sort by timestamp
        user_data.sort(key=lambda x: x['timestamp'])
        
        # Get the latest timestamp
        latest_time = user_data[-1]['timestamp']
        
        # Create primary window (most recent 45s)
        primary_start = latest_time - primary_window
        primary_events = [e for e in user_data if e['timestamp'] > primary_start]
        
        # Create secondary window (3min baseline before primary window)
        secondary_start = primary_start - secondary_window
        secondary_events = [e for e in user_data if secondary_start < e['timestamp'] <= primary_start]
        
        if len(primary_events) >= 10 and len(secondary_events) >= 20:
            windows[user_id] = {
                'primary': primary_events,
                'secondary': secondary_events
            }
    
    print(f"Created time windows for {len(windows)} users")
    return windows

def extract_features_from_window(events):
    """Extract keystroke timing features from a window of events"""
    if not events:
        return {
            'dwell_times': np.array([]),
            'intervals': np.array([])
        }
    
    # Sort by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    # Extract key down and key up events
    key_down_events = [e for e in events if e['event_type'] == 'KD']
    key_up_events = [e for e in events if e['event_type'] == 'KU']
    
    # Calculate dwell times (key hold durations)
    dwell_times = []
    for down_event in key_down_events:
        key = down_event['key']
        down_time = down_event['timestamp']
        
        # Find matching key up event
        for up_event in key_up_events:
            if up_event['key'] == key and up_event['timestamp'] > down_time:
                dwell_time = up_event['timestamp'] - down_time
                if 10 < dwell_time < 1000:  # Reasonable range: 10ms to 1s
                    dwell_times.append(dwell_time)
                break
    
    # Calculate inter-key intervals
    intervals = []
    for i in range(1, len(key_down_events)):
        interval = key_down_events[i]['timestamp'] - key_down_events[i-1]['timestamp']
        if 10 < interval < 2000:  # Reasonable range: 10ms to 2s
            intervals.append(interval)
    
    return {
        'dwell_times': np.array(dwell_times),
        'intervals': np.array(intervals)
    }

def calculate_similarity(primary_features, secondary_features):
    """
    Calculate similarity between primary and secondary window features.
    Returns a value between 0 and 1, where higher values indicate more similarity.
    """
    similarities = []
    
    # Compare dwell time distributions
    if len(primary_features['dwell_times']) >= 5 and len(secondary_features['dwell_times']) >= 5:
        # Use Kolmogorov-Smirnov test to compare distributions
        # Lower p-value means distributions are different
        try:
            _, p_value = ks_2samp(primary_features['dwell_times'], secondary_features['dwell_times'])
            similarities.append(p_value)  # Higher p-value = more similar distributions
        except:
            similarities.append(0.5)  # Default if test fails
    
    # Compare interval distributions
    if len(primary_features['intervals']) >= 5 and len(secondary_features['intervals']) >= 5:
        try:
            _, p_value = ks_2samp(primary_features['intervals'], secondary_features['intervals'])
            similarities.append(p_value)
        except:
            similarities.append(0.5)
    
    # If no comparisons were made, return 0.5 (neutral)
    if not similarities:
        return 0.5
    
    # Return average similarity
    return np.mean(similarities)

def evaluate_users(windows):
    """Evaluate all users and return similarity scores"""
    print("Evaluating typing pattern similarity...")
    
    results = {}
    
    for user_id, user_windows in windows.items():
        # Extract features from each window
        primary_features = extract_features_from_window(user_windows['primary'])
        secondary_features = extract_features_from_window(user_windows['secondary'])
        
        # Calculate similarity
        similarity = calculate_similarity(primary_features, secondary_features)
        
        results[user_id] = similarity
    
    return results

def split_data(events, test_ratio=0.2):
    """Split data into training and testing sets based on users"""
    # Group by user
    user_events = defaultdict(list)
    for event in events:
        user_events[event['user_id']].append(event)
    
    # Get unique user IDs
    user_ids = list(user_events.keys())
    random.shuffle(user_ids)
    
    # Split users for train/test
    split_idx = int(len(user_ids) * (1 - test_ratio))
    train_users = user_ids[:split_idx]
    test_users = user_ids[split_idx:]
    
    # Create event lists
    train_events = [e for e in events if e['user_id'] in train_users]
    test_events = [e for e in events if e['user_id'] in test_users]
    
    print(f"Split data: {len(train_events)} training events, {len(test_events)} testing events")
    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    
    return train_events, test_events

def train_model(sincere_train, cheating_train):
    """
    'Train' the model by determining the optimal similarity threshold.
    This is a simple approach that finds a threshold that separates sincere and cheating users.
    """
    print("Training model...")
    
    # Create time windows
    sincere_windows = create_time_windows(sincere_train)
    cheating_windows = create_time_windows(cheating_train)
    
    # Calculate similarities
    sincere_similarities = evaluate_users(sincere_windows)
    cheating_similarities = evaluate_users(cheating_windows)
    
    # Get similarity values
    sincere_values = list(sincere_similarities.values())
    cheating_values = list(cheating_similarities.values())
    
    if not sincere_values or not cheating_values:
        print("Error: Not enough data to train model")
        return 0.5  # Default threshold
    
    # Calculate average similarities
    sincere_avg = np.mean(sincere_values)
    cheating_avg = np.mean(cheating_values)
    
    print(f"Average similarity - Sincere: {sincere_avg:.4f}, Cheating: {cheating_avg:.4f}")
    
    # Try different thresholds to find optimal one
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_accuracy = 0.0
    
    for threshold in thresholds:
        sincere_correct = sum(s >= threshold for s in sincere_values) / len(sincere_values) if sincere_values else 0
        cheating_correct = sum(s < threshold for s in cheating_values) / len(cheating_values) if cheating_values else 0
        accuracy = (sincere_correct + cheating_correct) / 2
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.4f} (training accuracy: {best_accuracy:.4f})")
    
    # Plot similarity distributions
    plt.figure(figsize=(10, 6))
    plt.hist(sincere_values, bins=20, alpha=0.5, label='Sincere', color='green')
    plt.hist(cheating_values, bins=20, alpha=0.5, label='Cheating', color='red')
    plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Threshold: {best_threshold:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('similarity_distribution.png')
    print("Saved similarity distribution plot")
    
    return best_threshold

def evaluate_model(threshold, sincere_test, cheating_test):
    """Evaluate model performance using the provided threshold"""
    print("Evaluating model performance...")
    
    # Create time windows
    sincere_windows = create_time_windows(sincere_test)
    cheating_windows = create_time_windows(cheating_test)
    
    # Calculate similarities
    sincere_similarities = evaluate_users(sincere_windows)
    cheating_similarities = evaluate_users(cheating_windows)
    
    # Get similarity values
    sincere_values = list(sincere_similarities.values())
    cheating_values = list(cheating_similarities.values())
    
    if not sincere_values or not cheating_values:
        print("Error: Not enough test data")
        return {
            'sincere_acc': 0.0,
            'cheating_acc': 0.0,
            'overall_acc': 0.0
        }
    
    # Calculate accuracy
    sincere_correct = sum(s >= threshold for s in sincere_values) / len(sincere_values)
    cheating_correct = sum(s < threshold for s in cheating_values) / len(cheating_values)
    overall_acc = (sincere_correct + cheating_correct) / 2
    
    # Display results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Sincere average similarity: {np.mean(sincere_values):.4f}")
    print(f"Cheating average similarity: {np.mean(cheating_values):.4f}")
    print(f"Decision threshold: {threshold:.4f}")
    print("\nAccuracy:")
    print(f"  Sincere: {sincere_correct:.2f}")
    print(f"  Cheating: {cheating_correct:.2f}")
    print(f"  Overall: {overall_acc:.2f}")
    print("="*50 + "\n")
    
    return {
        'sincere_acc': sincere_correct,
        'cheating_acc': cheating_correct,
        'overall_acc': overall_acc,
        'sincere_similarity': np.mean(sincere_values),
        'cheating_similarity': np.mean(cheating_values),
        'threshold': threshold
    }

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    sincere_events = load_keystroke_data(args.sincere)
    cheating_events = load_keystroke_data(args.cheating)
    
    # Split data
    sincere_train, sincere_test = split_data(sincere_events, args.test_split)
    cheating_train, cheating_test = split_data(cheating_events, args.test_split)
    
    # Train model (find optimal threshold)
    start_time = time.time()
    threshold = train_model(sincere_train, cheating_train)
    train_time = time.time() - start_time
    
    # Evaluate model
    start_time = time.time()
    results = evaluate_model(threshold, sincere_test, cheating_test)
    eval_time = time.time() - start_time
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print(f"Sincere accuracy: {results['sincere_acc']:.2f}")
    print(f"Cheating accuracy: {results['cheating_acc']:.2f}")
    print(f"Overall accuracy: {results['overall_acc']:.2f}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Save results
    with open(os.path.join(args.output_dir, "comparison_results.txt"), "w") as f:
        f.write(f"Sincere accuracy: {results['sincere_acc']:.2f}\n")
        f.write(f"Cheating accuracy: {results['cheating_acc']:.2f}\n")
        f.write(f"Overall accuracy: {results['overall_acc']:.2f}\n")
        f.write(f"Sincere similarity: {results['sincere_similarity']:.4f}\n")
        f.write(f"Cheating similarity: {results['cheating_similarity']:.4f}\n")
        f.write(f"Decision threshold: {results['threshold']:.4f}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write(f"Evaluation time: {eval_time:.2f} seconds\n")
    
    print(f"Results saved to {os.path.join(args.output_dir, 'comparison_results.txt')}")

if __name__ == "__main__":
    main() 