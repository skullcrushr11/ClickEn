#!/usr/bin/env python3
import os
import argparse
import time
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import pandas as pd

from keystroke_hmm_optimized import ProctoringSystem, load_keystroke_data, STATES, extract_user_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Keystroke Proctoring System Demo')
    parser.add_argument('--data', type=str, default='sincere.json',
                        help='Path to the keystroke data JSON file (default: sincere.json)')
    parser.add_argument('--sincere', type=str, default='sincere.json',
                        help='Path to sincere data for pretraining (default: sincere.json)')
    parser.add_argument('--cheating', type=str, default='cheating.json',
                        help='Path to cheating data for pretraining (default: cheating.json)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable real-time plotting')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Disable pretraining with labeled data')
    parser.add_argument('--user-id', type=str, default=None,
                        help='Specific user ID to monitor (default: None, uses all users)')
    return parser.parse_args()

def simulate_keystroke_stream(keystroke_data, speed_multiplier=1.0, user_id=None):
    """
    Simulate a stream of keystrokes by yielding them at appropriate times
    
    Args:
        keystroke_data: List of keystroke events
        speed_multiplier: Speed multiplier for simulation (default: 1.0)
        user_id: Specific user ID to monitor (default: None)
    """
    if not keystroke_data:
        print("No keystroke data to simulate")
        return
    
    # Filter by user_id if specified
    if user_id:
        keystroke_data = [k for k in keystroke_data if k.get('user_id') == user_id]
        if not keystroke_data:
            print(f"No data found for user_id: {user_id}")
            return
        print(f"Filtered to {len(keystroke_data)} events for user_id: {user_id}")
    
    # Start from the first timestamp
    start_time = keystroke_data[0]['timestamp']
    real_start_time = datetime.now()
    
    for i, keystroke in enumerate(keystroke_data):
        # Calculate when to yield this keystroke
        elapsed_real_time = (datetime.now() - real_start_time).total_seconds()
        simulation_time = (keystroke['timestamp'] - start_time).total_seconds() / speed_multiplier
        
        # Wait if we're ahead of the simulation time
        if elapsed_real_time < simulation_time:
            time.sleep(simulation_time - elapsed_real_time)
        
        # Yield the keystroke
        yield keystroke
        
        # Print progress every 100 keystrokes
        if i % 100 == 0:
            print(f"Processed {i}/{len(keystroke_data)} keystrokes "
                  f"({i/len(keystroke_data)*100:.1f}%)")

def setup_visualization():
    """Set up the real-time visualization"""
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Keystroke Proctoring System', fontsize=16)
    
    # Initialize data structures for plotting
    window_size = 100
    timestamps = deque(maxlen=window_size)
    risk_scores = deque(maxlen=window_size)
    states = deque(maxlen=window_size)
    
    # Create initial empty plot lines
    line1, = ax1.plot([], [], 'b-', label='Risk Score')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, window_size)
    ax1.set_title('Risk Score Over Time')
    ax1.set_ylabel('Risk Score')
    ax1.set_xlabel('Time')
    ax1.grid(True)
    ax1.legend()
    
    # Create state plot
    line2, = ax2.plot([], [], 'r-', label='State')
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_xlim(0, window_size)
    ax2.set_title('State Classification Over Time')
    ax2.set_ylabel('State')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(list(STATES.values()))
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, (ax1, ax2), (line1, line2), (timestamps, risk_scores, states)

def update_plot(frame, lines, data_deques, system):
    """Update function for the animation"""
    line1, line2 = lines
    timestamps, risk_scores, states = data_deques
    
    # Get current status
    status = system.get_status()
    
    # Add new data points
    timestamps.append(len(timestamps))
    risk_scores.append(status['risk_score'])
    states.append(status['state_code'])
    
    # Update data in the plot
    x_data = list(range(len(timestamps)))
    line1.set_data(x_data, list(risk_scores))
    line2.set_data(x_data, list(states))
    
    # Update xlim for scrolling effect if we have enough data
    if len(timestamps) >= 100:
        line1.axes.set_xlim(len(timestamps) - 100, len(timestamps))
        line2.axes.set_xlim(len(timestamps) - 100, len(timestamps))
    
    # Update plot title with current state and risk score
    line1.axes.set_title(f'Risk Score: {status["risk_score"]:.2f}')
    line2.axes.set_title(f'Current State: {status["state"]}')
    
    return line1, line2

def export_results(system, output_file='proctoring_results.csv'):
    """Export the proctoring results to a CSV file"""
    if not system.feature_history or not system.state_history:
        print("No data to export")
        return
    
    # Create DataFrame from features
    feature_df = pd.DataFrame(system.feature_history)
    
    # Add state and risk score
    feature_df['state'] = pd.Series(system.state_history)
    feature_df['risk_score'] = pd.Series(system.risk_score_history)
    feature_df['state_name'] = feature_df['state'].map(STATES)
    
    # Save to CSV
    feature_df.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")
    
    # Generate summary statistics
    summary = {
        'total_keystrokes': len(system.keystroke_processor.baseline_window),
        'average_risk_score': np.mean(system.risk_score_history),
        'max_risk_score': np.max(system.risk_score_history),
        'state_distribution': {
            state_name: (np.array(system.state_history) == state_code).sum() / len(system.state_history) * 100
            for state_code, state_name in STATES.items()
        }
    }
    
    # Print summary
    print("\nSummary:")
    print(f"Total keystrokes processed: {summary['total_keystrokes']}")
    print(f"Average risk score: {summary['average_risk_score']:.2f}")
    print(f"Maximum risk score: {summary['max_risk_score']:.2f}")
    print("\nState distribution:")
    for state, percentage in summary['state_distribution'].items():
        print(f"  {state}: {percentage:.1f}%")

def main():
    """Main function for the demo"""
    args = parse_arguments()
    
    # Create proctoring system with pretraining if enabled
    if args.no_pretrain:
        print("Pretraining disabled, using real-time training")
        system = ProctoringSystem(pretrained=False)
    else:
        print("Creating pretrained proctoring system")
        system = ProctoringSystem(pretrained=True)
        # Pretrain with both sincere and cheating data
        system.pretrain_with_datasets(args.sincere, args.cheating)
    
    # Load keystroke data for demo
    print(f"Loading keystroke data from {args.data}...")
    keystroke_data = load_keystroke_data(args.data)
    print(f"Loaded {len(keystroke_data)} keystroke events")
    
    # List available users if user_id not specified
    if not args.user_id:
        user_data = extract_user_data(keystroke_data)
        user_ids = list(user_data.keys())
        print(f"Found {len(user_ids)} users in the dataset. User IDs: {', '.join(user_ids[:5])}" + 
              (f" and {len(user_ids)-5} more..." if len(user_ids) > 5 else ""))
    else:
        # Set the user ID for the system
        system.set_user_id(args.user_id)
    
    # Set up visualization if enabled
    if not args.no_plot:
        fig, axes, lines, data_deques = setup_visualization()
        ani = FuncAnimation(fig, update_plot, fargs=(lines, data_deques, system), 
                           interval=100, blit=True)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
    
    # Process keystroke events
    print(f"Starting simulation at {args.speed}x speed...")
    for keystroke in simulate_keystroke_stream(keystroke_data, args.speed, args.user_id):
        state, risk_score = system.process_keystroke(keystroke)
        
        # Update visualization
        if not args.no_plot:
            plt.pause(0.001)  # Small pause to update the plot
    
    # Keep the plot open if visualization is enabled
    if not args.no_plot:
        plt.ioff()
        plt.show()
    
    # Export results
    output_file = 'proctoring_results.csv'
    if args.user_id:
        output_file = f'proctoring_results_{args.user_id}.csv'
    export_results(system, output_file)

if __name__ == "__main__":
    main() 