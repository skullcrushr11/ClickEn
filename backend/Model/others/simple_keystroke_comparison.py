#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from collections import defaultdict
from scipy.spatial.distance import cdist

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
    parser.add_argument('--primary-window', type=int, default=45,
                      help='Primary window length in seconds')
    parser.add_argument('--secondary-window', type=int, default=180,
                      help='Secondary window length in seconds')
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

def extract_features(events):
    """
    Extract comprehensive feature vector from keystroke events.
    This creates a numerical representation of typing patterns.
    """
    if not events or len(events) < 10:
        return None
        
    # Sort by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    # Extract key down and key up events
    key_down_events = [e for e in events if e['event_type'] == 'KD']
    key_up_events = [e for e in events if e['event_type'] == 'KU']
    
    if len(key_down_events) < 10:
        return None
        
    # ------ Timing Features ------
    
    # 1. Inter-key intervals (time between consecutive key presses)
    intervals = []
    for i in range(1, len(key_down_events)):
        interval = key_down_events[i]['timestamp'] - key_down_events[i-1]['timestamp']
        if 10 < interval < 2000:  # Reasonable range
            intervals.append(interval)
    
    if not intervals:
        return None
        
    # 2. Key hold durations (time between key down and key up)
    durations = []
    key_to_timestamp = {}
    for e in key_down_events:
        key_to_timestamp[e['key']] = e['timestamp']
    
    for e in key_up_events:
        if e['key'] in key_to_timestamp:
            duration = e['timestamp'] - key_to_timestamp[e['key']]
            if 10 < duration < 500:  # Reasonable range
                durations.append(duration)
            # Remove to handle key repeats correctly
            del key_to_timestamp[e['key']]
    
    # 3. Flight times (time between key release and next key press)
    flight_times = []
    up_times = sorted([(e['key'], e['timestamp']) for e in key_up_events], key=lambda x: x[1])
    down_times = sorted([(e['key'], e['timestamp']) for e in key_down_events], key=lambda x: x[1])
    
    for i in range(len(up_times)):
        for j in range(len(down_times)):
            if down_times[j][1] > up_times[i][1]:
                flight = down_times[j][1] - up_times[i][1]
                if 5 < flight < 1000:  # Reasonable range
                    flight_times.append(flight)
                break
    
    # ------ Statistical Features ------
    
    # For each timing dimension, extract statistical features
    feature_vector = []
    
    # Add interval statistics
    if intervals:
        feature_vector.extend([
            np.mean(intervals),
            np.median(intervals),
            np.std(intervals) if len(intervals) > 1 else 0,
            np.percentile(intervals, 25) if len(intervals) >= 4 else np.min(intervals),
            np.percentile(intervals, 75) if len(intervals) >= 4 else np.max(intervals)
        ])
    else:
        feature_vector.extend([0, 0, 0, 0, 0])
    
    # Add duration statistics
    if durations:
        feature_vector.extend([
            np.mean(durations),
            np.median(durations),
            np.std(durations) if len(durations) > 1 else 0,
            np.percentile(durations, 25) if len(durations) >= 4 else np.min(durations),
            np.percentile(durations, 75) if len(durations) >= 4 else np.max(durations)
        ])
    else:
        feature_vector.extend([0, 0, 0, 0, 0])
    
    # Add flight time statistics
    if flight_times:
        feature_vector.extend([
            np.mean(flight_times),
            np.median(flight_times),
            np.std(flight_times) if len(flight_times) > 1 else 0
        ])
    else:
        feature_vector.extend([0, 0, 0])
    
    # ------ Pattern Features ------
    
    # Key frequency distribution (top 10 most common keys)
    key_counts = defaultdict(int)
    for e in key_down_events:
        key_counts[e['key']] += 1
    
    # Normalize to get frequencies
    total_keys = sum(key_counts.values())
    if total_keys > 0:
        key_freqs = {k: count/total_keys for k, count in key_counts.items()}
        
        # Get frequencies of the 10 most common keys
        top_keys = sorted(key_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_vector.extend([freq for _, freq in top_keys])
        # Pad if less than 10 keys
        feature_vector.extend([0] * (10 - len(top_keys)))
    else:
        feature_vector.extend([0] * 10)
    
    # Add typing pace features
    total_time = events[-1]['timestamp'] - events[0]['timestamp']
    if total_time > 0:
        chars_per_second = len(key_down_events) / (total_time / 1000)
        feature_vector.append(chars_per_second)
    else:
        feature_vector.append(0)
    
    # Add interval variance features
    if intervals and len(intervals) > 5:
        # Calculate variance in different segments to measure rhythm consistency
        half_point = len(intervals) // 2
        var_first_half = np.var(intervals[:half_point]) if half_point > 1 else 0
        var_second_half = np.var(intervals[half_point:]) if len(intervals) - half_point > 1 else 0
        feature_vector.extend([var_first_half, var_second_half])
    else:
        feature_vector.extend([0, 0])
    
    return np.array(feature_vector)

def calculate_similarity(primary_window, secondary_window):
    """
    Calculate similarity between primary and secondary windows
    based on feature vectors and pattern matching.
    """
    # Extract comprehensive feature vectors
    primary_features = extract_features(primary_window)
    secondary_features = extract_features(secondary_window)
    
    if primary_features is None or secondary_features is None:
        return 0.5  # Not enough data
    
    # Calculate cosine similarity between feature vectors
    cosine_sim = cosine_similarity([primary_features], [secondary_features])[0][0]
    
    # Calculate normalized Euclidean distance
    euclidean_dist = np.linalg.norm(primary_features - secondary_features)
    max_dist = np.sqrt(len(primary_features))  # Maximum possible distance
    euclidean_sim = 1.0 - (euclidean_dist / max_dist)
    
    # Calculate correlation between vectors
    try:
        correlation, _ = pearsonr(primary_features, secondary_features)
        if np.isnan(correlation):
            correlation = 0
        correlation_sim = (correlation + 1) / 2  # Convert from [-1,1] to [0,1]
    except:
        correlation_sim = 0.5
    
    # Combine all similarity metrics
    # Weighted combination based on which metrics are typically more reliable
    similarity = 0.5 * cosine_sim + 0.3 * euclidean_sim + 0.2 * correlation_sim
    
    return similarity

def evaluate_users(windows):
    """Evaluate all users and return similarity scores"""
    print("Evaluating typing pattern similarity...")
    
    results = {}
    
    for user_id, user_windows in windows.items():
        # Calculate similarity between primary and secondary windows
        similarity = calculate_similarity(user_windows['primary'], user_windows['secondary'])
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
    """Train model by finding optimal threshold between sincere and cheating typing patterns"""
    print("Training model...")
    
    # Create time windows
    sincere_windows = create_time_windows(sincere_train)
    cheating_windows = create_time_windows(cheating_train)
    
    # Calculate similarity scores
    sincere_similarities = evaluate_users(sincere_windows)
    cheating_similarities = evaluate_users(cheating_windows)
    
    # Get similarity values
    sincere_values = list(sincere_similarities.values())
    cheating_values = list(cheating_similarities.values())
    
    if not sincere_values or not cheating_values:
        print("Error: Not enough data to train model")
        return 0.5
    
    # Calculate average similarities for debugging
    sincere_avg = np.mean(sincere_values)
    cheating_avg = np.mean(cheating_values)
    
    print(f"Average similarity - Sincere: {sincere_avg:.4f}, Cheating: {cheating_avg:.4f}")
    print(f"Sincere range: {min(sincere_values):.4f} - {max(sincere_values):.4f}")
    print(f"Cheating range: {min(cheating_values):.4f} - {max(cheating_values):.4f}")
    
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
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = np.linspace(0, 1, 20)
    plt.hist(sincere_values, bins=bins, alpha=0.5, label='Sincere', color='green')
    plt.hist(cheating_values, bins=bins, alpha=0.5, label='Cheating', color='red')
    
    # Add threshold line
    plt.axvline(x=best_threshold, color='black', linestyle='--', 
                label=f'Threshold: {best_threshold:.3f}')
    
    # Add labels
    plt.xlabel('Similarity Score (higher = more consistent typing)')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Typing Pattern Similarity Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add accuracy information
    plt.figtext(0.02, 0.02, 
                f"Training Accuracy: {best_accuracy:.2f}\n"
                f"Sincere Accuracy: {sincere_correct:.2f}\n"
                f"Cheating Accuracy: {cheating_correct:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('similarity_distribution.png')
    print("Saved similarity distribution plot")
    
    return best_threshold

def evaluate_model(threshold, sincere_test, cheating_test):
    """Evaluate model performance using the provided threshold"""
    print("Evaluating model performance...")
    
    # Create time windows
    sincere_windows = create_time_windows(sincere_test)
    cheating_windows = create_time_windows(cheating_test)
    
    if len(sincere_windows) == 0 or len(cheating_windows) == 0:
        print("Warning: Not enough test data with adequate windows")
    
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
    
    # Convert window sizes to milliseconds
    primary_window_ms = args.primary_window * 1000
    secondary_window_ms = args.secondary_window * 1000
    
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
    print("\nFINAL RESULTS:")
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