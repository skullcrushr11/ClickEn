#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from keystroke_hmm_optimized import ProctoringSystem, load_keystroke_data, STATES, extract_user_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Keystroke Proctoring System')
    parser.add_argument('--sincere', type=str, default='sincere.json',
                        help='Path to sincere keystroke data (default: sincere.json)')
    parser.add_argument('--cheating', type=str, default='cheating.json',
                        help='Path to cheating keystroke data (default: cheating.json)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Disable pretraining with labeled data')
    parser.add_argument('--user-ids', type=str, default=None,
                        help='Specific user IDs to evaluate as comma-separated list (default: None, evaluate all)')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Perform cross-validation by user (default: False)')
    return parser.parse_args()

def process_data_file(file_path, expected_label, system=None, user_filter=None):
    """
    Process a data file and evaluate the system's performance
    
    Args:
        file_path: Path to the keystroke data file
        expected_label: Expected classification label (0-3)
        system: ProctoringSystem instance (default: None, will create a new one)
        user_filter: Optional list of user IDs to filter (default: None)
        
    Returns:
        ProctoringSystem instance, performance metrics
    """
    print(f"Processing {file_path}...")
    
    # Load keystroke data
    keystroke_data = load_keystroke_data(file_path)
    print(f"Loaded {len(keystroke_data)} keystroke events")
    
    # Filter by user_id if specified
    if user_filter:
        keystroke_data = [k for k in keystroke_data if k.get('user_id') in user_filter]
        print(f"Filtered to {len(keystroke_data)} events for specified users")
    
    # Create or use existing proctoring system
    if system is None:
        system = ProctoringSystem()
    
    # Process keystroke events
    state_history = []
    risk_score_history = []
    
    # Group by user_id to process each user's data separately
    user_data = extract_user_data(keystroke_data)
    
    print(f"Processing data for {len(user_data)} users...")
    for user_id, user_keystrokes in user_data.items():
        # Set user ID in system
        system.set_user_id(user_id)
        
        # Reset keystrokes processor for each user
        system.keystroke_processor = system.keystroke_processor.__class__()
        
        print(f"Processing {len(user_keystrokes)} keystrokes for user {user_id}")
        for i, keystroke in enumerate(user_keystrokes):
            state, risk_score = system.process_keystroke(keystroke)
            state_history.append(state)
            risk_score_history.append(risk_score)
            
            # Print progress every 1000 keystrokes
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(user_keystrokes)} keystrokes "
                      f"({i/len(user_keystrokes)*100:.1f}%)")
    
    # Calculate performance metrics
    y_true = [expected_label] * len(state_history)
    y_pred = state_history
    
    # Basic metrics
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    
    # Average risk score
    avg_risk_score = np.mean(risk_score_history)
    
    # State distribution
    state_counts = {}
    for state_code in range(4):
        state_counts[STATES[state_code]] = np.sum(np.array(y_pred) == state_code) / len(y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'avg_risk_score': avg_risk_score,
        'state_distribution': state_counts,
        'expected_label': expected_label,
        'y_pred': y_pred,
        'y_true': y_true,
        'risk_scores': risk_score_history,
        'n_users': len(user_data)
    }
    
    return system, metrics

def cross_validate_by_user(sincere_path, cheating_path, output_dir):
    """
    Perform cross-validation by leaving out each user
    
    Args:
        sincere_path: Path to sincere data
        cheating_path: Path to cheating data
        output_dir: Output directory for results
    
    Returns:
        List of user-specific metrics
    """
    print("Performing cross-validation by user...")
    
    # Load all data
    sincere_data = load_keystroke_data(sincere_path)
    cheating_data = load_keystroke_data(cheating_path)
    
    # Extract user lists
    sincere_users = list(set(k.get('user_id') for k in sincere_data))
    cheating_users = list(set(k.get('user_id') for k in cheating_data))
    
    print(f"Found {len(sincere_users)} users in sincere data and {len(cheating_users)} users in cheating data")
    
    # Store results for each user
    user_metrics = []
    
    # Cross-validate for sincere users
    for test_user in sincere_users:
        print(f"\nCross-validating with test user {test_user} (sincere)...")
        
        # Create system and pretrain on all users except test user
        train_users = [u for u in sincere_users if u != test_user] + cheating_users
        system = create_pretrained_system(sincere_path, cheating_path, train_users)
        
        # Test on the excluded user
        test_system, metrics = process_data_file(sincere_path, 0, system, [test_user])
        
        metrics['user_id'] = test_user
        metrics['data_type'] = 'sincere'
        user_metrics.append(metrics)
    
    # Cross-validate for cheating users
    for test_user in cheating_users:
        print(f"\nCross-validating with test user {test_user} (cheating)...")
        
        # Create system and pretrain on all users except test user
        train_users = sincere_users + [u for u in cheating_users if u != test_user]
        system = create_pretrained_system(sincere_path, cheating_path, train_users)
        
        # Test on the excluded user
        test_system, metrics = process_data_file(cheating_path, 3, system, [test_user])
        
        metrics['user_id'] = test_user
        metrics['data_type'] = 'cheating'
        user_metrics.append(metrics)
    
    # Generate summary report
    generate_cross_validation_report(user_metrics, output_dir)
    
    return user_metrics

def create_pretrained_system(sincere_path, cheating_path, train_users=None):
    """
    Create a pretrained system using specified users
    
    Args:
        sincere_path: Path to sincere data
        cheating_path: Path to cheating data
        train_users: Optional list of user IDs to train on (default: None, train on all)
        
    Returns:
        Pretrained ProctoringSystem instance
    """
    system = ProctoringSystem(pretrained=True)
    
    # Load data
    sincere_data = load_keystroke_data(sincere_path)
    cheating_data = load_keystroke_data(cheating_path)
    
    # Filter by user_id if specified
    if train_users:
        sincere_data = [k for k in sincere_data if k.get('user_id') in train_users]
        cheating_data = [k for k in cheating_data if k.get('user_id') in train_users]
    
    # Process sincere data
    sincere_features = []
    processor = system.keystroke_processor.__class__()
    
    for keystroke in sincere_data:
        processor.add_keystroke(keystroke)
        features = processor.get_features()
        if features:
            sincere_features.append(features)
        
        # Reset processor after a window of events to handle multiple users
        if len(sincere_features) % 100 == 0:
            processor = system.keystroke_processor.__class__()
    
    # Process cheating data
    cheating_features = []
    processor = system.keystroke_processor.__class__()
    
    for keystroke in cheating_data:
        processor.add_keystroke(keystroke)
        features = processor.get_features()
        if features:
            cheating_features.append(features)
        
        # Reset processor after a window of events to handle multiple users
        if len(cheating_features) % 100 == 0:
            processor = system.keystroke_processor.__class__()
    
    # Combine features and create labels
    all_features = sincere_features + cheating_features
    labels = [0] * len(sincere_features) + [3] * len(cheating_features)
    
    # Train the HMM model
    system.hmm.train(all_features, labels)
    system.pretrained = True
    
    return system

def generate_confusion_matrix(metrics_list, output_dir):
    """
    Generate and save a confusion matrix
    
    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Output directory
    """
    # Combine predictions and true labels
    y_true = []
    y_pred = []
    
    for metrics in metrics_list:
        y_true.extend(metrics['y_true'])
        y_pred.extend(metrics['y_pred'])
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(STATES.values()),
                yticklabels=list(STATES.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def generate_classification_report(metrics_list, output_dir):
    """
    Generate and save a classification report
    
    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Output directory
    """
    # Combine predictions and true labels
    y_true = []
    y_pred = []
    
    for metrics in metrics_list:
        y_true.extend(metrics['y_true'])
        y_pred.extend(metrics['y_pred'])
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=list(STATES.values()))
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Print report
    print("\nClassification Report:")
    print(report)

def generate_risk_score_distribution(metrics_list, output_dir):
    """
    Generate and save risk score distribution plots
    
    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Output directory
    """
    fig, ax = plt.subplots(len(metrics_list), 1, figsize=(10, 6*len(metrics_list)))
    
    for i, metrics in enumerate(metrics_list):
        label = STATES[metrics['expected_label']]
        risk_scores = metrics['risk_scores']
        
        if len(metrics_list) == 1:
            axis = ax
        else:
            axis = ax[i]
        
        # Plot histogram of risk scores
        axis.hist(risk_scores, bins=20, alpha=0.7)
        axis.set_title(f'Risk Score Distribution - {label} ({metrics["n_users"]} users)')
        axis.set_xlabel('Risk Score')
        axis.set_ylabel('Frequency')
        axis.grid(True, alpha=0.3)
        
        # Add vertical lines for mean and thresholds
        axis.axvline(np.mean(risk_scores), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(risk_scores):.2f}')
        axis.axvline(0.3, color='g', linestyle=':', 
                     label='Not Cheating Threshold')
        axis.axvline(0.5, color='y', linestyle=':', 
                     label='Distracted Threshold')
        axis.axvline(0.8, color='r', linestyle=':', 
                     label='Maybe Cheating Threshold')
        axis.legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'risk_score_distribution.png'))
    plt.close()

def generate_feature_analysis(metrics_list, system, output_dir):
    """
    Generate and save feature analysis plots
    
    Args:
        metrics_list: List of metrics dictionaries
        system: ProctoringSystem instance
        output_dir: Output directory
    """
    # Get feature history
    feature_df = pd.DataFrame(system.feature_history)
    
    # Add risk score and state
    feature_df['risk_score'] = pd.Series(system.risk_score_history)
    feature_df['state'] = pd.Series(system.state_history)
    feature_df['state_name'] = feature_df['state'].map(STATES)
    
    # Save features to CSV
    os.makedirs(output_dir, exist_ok=True)
    feature_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    corr = feature_df[numeric_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    plt.close()
    
    # Create feature distribution by state
    feature_names = ['iki_mean', 'hold_mean', 'special_keys_freq', 'dtw_distance', 
                     'typing_speed', 'iki_mean_deviation', 'hold_mean_deviation']
    
    for feature_name in feature_names:
        plt.figure(figsize=(10, 6))
        for state_code in range(4):
            state_data = feature_df[feature_df['state'] == state_code][feature_name]
            if len(state_data) > 0:
                sns.kdeplot(state_data, label=STATES[state_code])
        
        plt.title(f'Distribution of {feature_name} by State')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'feature_dist_{feature_name}.png'))
        plt.close()

def generate_cross_validation_report(user_metrics, output_dir):
    """
    Generate a report summarizing cross-validation results
    
    Args:
        user_metrics: List of metrics dictionaries from cross-validation
        output_dir: Output directory
    """
    # Create summary table
    summary_data = []
    for metrics in user_metrics:
        summary_data.append({
            'user_id': metrics['user_id'],
            'data_type': metrics['data_type'],
            'accuracy': metrics['accuracy'],
            'avg_risk_score': metrics['avg_risk_score'],
            'state_distribution': str(metrics['state_distribution'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate average accuracy by data type
    sincere_acc = summary_df[summary_df['data_type'] == 'sincere']['accuracy'].mean()
    cheating_acc = summary_df[summary_df['data_type'] == 'cheating']['accuracy'].mean()
    overall_acc = summary_df['accuracy'].mean()
    
    # Save summary report
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'cross_validation_summary.csv'), index=False)
    
    # Create summary figure
    plt.figure(figsize=(12, 6))
    
    # Accuracy by user
    plt.subplot(1, 2, 1)
    sincere_users = summary_df[summary_df['data_type'] == 'sincere']
    cheating_users = summary_df[summary_df['data_type'] == 'cheating']
    
    plt.bar(range(len(sincere_users)), sincere_users['accuracy'], alpha=0.7, label='Sincere Users')
    plt.bar(range(len(sincere_users), len(sincere_users) + len(cheating_users)), 
            cheating_users['accuracy'], alpha=0.7, label='Cheating Users')
    
    plt.axhline(sincere_acc, color='blue', linestyle='--', label=f'Sincere Avg: {sincere_acc:.2f}')
    plt.axhline(cheating_acc, color='red', linestyle='--', label=f'Cheating Avg: {cheating_acc:.2f}')
    plt.axhline(overall_acc, color='green', linestyle='-', label=f'Overall Avg: {overall_acc:.2f}')
    
    plt.xlabel('Users')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy by User')
    plt.legend()
    
    # Risk scores by user type
    plt.subplot(1, 2, 2)
    plt.boxplot([
        summary_df[summary_df['data_type'] == 'sincere']['avg_risk_score'],
        summary_df[summary_df['data_type'] == 'cheating']['avg_risk_score']
    ], labels=['Sincere Users', 'Cheating Users'])
    plt.ylabel('Average Risk Score')
    plt.title('Risk Score Distribution by User Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'))
    plt.close()
    
    # Print summary
    print("\nCross-Validation Summary:")
    print(f"Overall Accuracy: {overall_acc:.2f}")
    print(f"Sincere Users Accuracy: {sincere_acc:.2f}")
    print(f"Cheating Users Accuracy: {cheating_acc:.2f}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Parse user IDs if provided
    user_filter = None
    if args.user_ids:
        user_filter = args.user_ids.split(',')
        print(f"Filtering to users: {user_filter}")
    
    # Perform cross-validation if requested
    if args.cross_validate:
        cross_validate_by_user(args.sincere, args.cheating, args.output)
        return
    
    # Initialize the system with pretraining if enabled
    if args.no_pretrain:
        print("Pretraining disabled, using real-time training")
        system = ProctoringSystem(pretrained=False)
    else:
        print("Creating pretrained proctoring system")
        system = ProctoringSystem(pretrained=True)
        # Pretrain with both sincere and cheating data
        system.pretrain_with_datasets(args.sincere, args.cheating)
    
    # Process sincere data
    system, sincere_metrics = process_data_file(args.sincere, 0, system, user_filter)
    
    # Process cheating data
    system, cheating_metrics = process_data_file(args.cheating, 3, system, user_filter)
    
    # List of all metrics
    metrics_list = [sincere_metrics, cheating_metrics]
    
    # Generate reports and visualizations
    generate_confusion_matrix(metrics_list, args.output)
    generate_classification_report(metrics_list, args.output)
    generate_risk_score_distribution(metrics_list, args.output)
    generate_feature_analysis(metrics_list, system, args.output)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Sincere Data - Accuracy: {sincere_metrics['accuracy']:.2f}, "
          f"Avg Risk Score: {sincere_metrics['avg_risk_score']:.2f}, "
          f"Users: {sincere_metrics['n_users']}")
    print(f"Cheating Data - Accuracy: {cheating_metrics['accuracy']:.2f}, "
          f"Avg Risk Score: {cheating_metrics['avg_risk_score']:.2f}, "
          f"Users: {cheating_metrics['n_users']}")
    
    print("\nSincere Data - State Distribution:")
    for state, percentage in sincere_metrics['state_distribution'].items():
        print(f"  {state}: {percentage*100:.1f}%")
    
    print("\nCheating Data - State Distribution:")
    for state, percentage in cheating_metrics['state_distribution'].items():
        print(f"  {state}: {percentage*100:.1f}%")

if __name__ == "__main__":
    main() 