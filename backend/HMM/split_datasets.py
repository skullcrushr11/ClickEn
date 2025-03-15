#!/usr/bin/env python3
import os
import sys
import json
import random
from datetime import datetime
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split keystroke datasets for training and testing')
    parser.add_argument('--sincere', type=str, required=True,
                      help='Path to sincere keystroke data')
    parser.add_argument('--cheating', type=str, required=True,
                      help='Path to cheating keystroke data')
    parser.add_argument('--test-split', type=float, default=0.2,
                      help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for split datasets')
    parser.add_argument('--by-user', action='store_true',
                      help='Split by user rather than by events')
    return parser.parse_args()

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def split_data_by_events(data, test_split):
    # Get all user IDs
    user_ids = list(data.keys())
    
    train_data = {}
    test_data = {}
    
    for user_id in user_ids:
        if "keyboard_data" in data[user_id]:
            # Shuffle the keyboard data for this user
            keyboard_events = data[user_id]["keyboard_data"].copy()
            random.shuffle(keyboard_events)
            
            # Calculate split point
            split_idx = int(len(keyboard_events) * (1 - test_split))
            
            # Split data
            train_events = keyboard_events[:split_idx]
            test_events = keyboard_events[split_idx:]
            
            # Create user data in train set
            train_data[user_id] = data[user_id].copy()
            train_data[user_id]["keyboard_data"] = train_events
            
            # Create user data in test set
            test_data[user_id] = data[user_id].copy()
            test_data[user_id]["keyboard_data"] = test_events
    
    return train_data, test_data

def split_data_by_users(data, test_split):
    # Get all user IDs
    user_ids = list(data.keys())
    
    # Shuffle the user IDs
    random.shuffle(user_ids)
    
    # Calculate split point
    split_idx = int(len(user_ids) * (1 - test_split))
    
    # Split user IDs
    train_users = user_ids[:split_idx]
    test_users = user_ids[split_idx:]
    
    # Create train and test data
    train_data = {user_id: data[user_id] for user_id in train_users}
    test_data = {user_id: data[user_id] for user_id in test_users}
    
    return train_data, test_data

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load data
    print(f"Loading sincere data from {args.sincere}...")
    sincere_data = load_data(args.sincere)
    
    print(f"Loading cheating data from {args.cheating}...")
    cheating_data = load_data(args.cheating)
    
    # Split data
    split_function = split_data_by_users if args.by_user else split_data_by_events
    
    print(f"Splitting sincere data with test ratio {args.test_split}...")
    sincere_train, sincere_test = split_function(sincere_data, args.test_split)
    
    print(f"Splitting cheating data with test ratio {args.test_split}...")
    cheating_train, cheating_test = split_function(cheating_data, args.test_split)
    
    # Save train data
    sincere_train_path = os.path.join(args.output_dir, "sincere_train.json")
    cheating_train_path = os.path.join(args.output_dir, "cheating_train.json")
    
    print(f"Saving train data to {sincere_train_path} and {cheating_train_path}...")
    save_data(sincere_train, sincere_train_path)
    save_data(cheating_train, cheating_train_path)
    
    # Save test data
    sincere_test_path = os.path.join(args.output_dir, "sincere_test.json")
    cheating_test_path = os.path.join(args.output_dir, "cheating_test.json")
    
    print(f"Saving test data to {sincere_test_path} and {cheating_test_path}...")
    save_data(sincere_test, sincere_test_path)
    save_data(cheating_test, cheating_test_path)
    
    # Print summary
    train_sincere_users = len(sincere_train)
    train_cheating_users = len(cheating_train)
    test_sincere_users = len(sincere_test)
    test_cheating_users = len(cheating_test)
    
    train_sincere_events = sum(len(user_data.get("keyboard_data", [])) for user_data in sincere_train.values())
    train_cheating_events = sum(len(user_data.get("keyboard_data", [])) for user_data in cheating_train.values())
    test_sincere_events = sum(len(user_data.get("keyboard_data", [])) for user_data in sincere_test.values())
    test_cheating_events = sum(len(user_data.get("keyboard_data", [])) for user_data in cheating_test.values())
    
    print("\nData Split Summary:")
    print(f"Training Set: {train_sincere_users} sincere users, {train_cheating_users} cheating users")
    print(f"             {train_sincere_events} sincere events, {train_cheating_events} cheating events")
    print(f"Testing Set: {test_sincere_users} sincere users, {test_cheating_users} cheating users")
    print(f"             {test_sincere_events} sincere events, {test_cheating_events} cheating events")
    
    # Return paths for the bash script
    print(f"SINCERE_TRAIN:{sincere_train_path}")
    print(f"CHEATING_TRAIN:{cheating_train_path}")
    print(f"SINCERE_TEST:{sincere_test_path}")
    print(f"CHEATING_TEST:{cheating_test_path}")

if __name__ == "__main__":
    main()
