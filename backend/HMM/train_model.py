#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# Import the right module based on whether we're using the optimized version
try:
    from keystroke_hmm_optimized import ProctoringSystem, load_keystroke_data, STATES, extract_user_data
    USING_OPTIMIZED = True
except ImportError:
    from keystroke_hmm import ProctoringSystem, load_keystroke_data, STATES, extract_user_data
    USING_OPTIMIZED = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate the keystroke proctoring model')
    parser.add_argument('--sincere-train', type=str, required=True,
                      help='Path to sincere training data')
    parser.add_argument('--cheating-train', type=str, required=True,
                      help='Path to cheating training data')
    parser.add_argument('--sincere-test', type=str, required=True,
                      help='Path to sincere test data')
    parser.add_argument('--cheating-test', type=str, required=True,
                      help='Path to cheating test data')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                      help='Display detailed output')
    return parser.parse_args()

def train_model(sincere_path, cheating_path, verbose=False):
    """
    Train a model on the provided datasets
    
    Args:
        sincere_path: Path to sincere data
        cheating_path: Path to cheating data
        verbose: Whether to display detailed output
        
    Returns:
        Trained ProctoringSystem
    """
    if verbose:
        print(f"Training model with data from {sincere_path} and {cheating_path}...")
    
    # Create a pretrained system
    system = ProctoringSystem(pretrained=True)
    
    # Train the model - use a larger sample size for better accuracy
    sample_size = 5000  # Increased from default
    success = system.pretrain_with_datasets(sincere_path, cheating_path, sample_size=sample_size)
    
    if not success:
        print("WARNING: Model training was not successful")
    
    return system

def evaluate_model(system, sincere_path, cheating_path, output_dir, verbose=False):
    """
    Evaluate the model on test data
    
    Args:
        system: Trained ProctoringSystem
        sincere_path: Path to sincere test data
        cheating_path: Path to cheating test data
        output_dir: Directory to save results
        verbose: Whether to display detailed output
        
    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print(f"Evaluating model on test data...")
    
    # Process sincere data
    sincere_results = process_test_data(system, sincere_path, 0, verbose)
    
    # Process cheating data
    cheating_results = process_test_data(system, cheating_path, 3, verbose)
    
    # Combine results
    all_predictions = sincere_results['predictions'] + cheating_results['predictions']
    all_true_labels = sincere_results['true_labels'] + cheating_results['true_labels']
    all_risk_scores = sincere_results['risk_scores'] + cheating_results['risk_scores']
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Calculate class-specific metrics
    class_metrics = {}
    for state_code, state_name in STATES.items():
        # For each class, calculate:
        # - True positives: prediction == state_code and true label == state_code
        # - False positives: prediction == state_code and true label != state_code
        # - False negatives: prediction != state_code and true label == state_code
        # - True negatives: prediction != state_code and true label != state_code
        
        tp = sum(1 for p, t in zip(all_predictions, all_true_labels) if p == state_code and t == state_code)
        fp = sum(1 for p, t in zip(all_predictions, all_true_labels) if p == state_code and t != state_code)
        fn = sum(1 for p, t in zip(all_predictions, all_true_labels) if p != state_code and t == state_code)
        tn = sum(1 for p, t in zip(all_predictions, all_true_labels) if p != state_code and t != state_code)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[state_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': tp + fn
        }
    
    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=list(STATES.values()), output_dict=True)
    
    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=list(STATES.values()),
              yticklabels=list(STATES.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Create risk score distribution plot
    plt.figure(figsize=(10, 6))
    
    # Split risk scores by true label
    risk_scores_by_class = {}
    for i, true_label in enumerate(all_true_labels):
        class_name = STATES[true_label]
        if class_name not in risk_scores_by_class:
            risk_scores_by_class[class_name] = []
        risk_scores_by_class[class_name].append(all_risk_scores[i])
    
    # Plot histogram for each class
    for class_name, scores in risk_scores_by_class.items():
        if scores:  # Only plot if we have scores
            sns.kdeplot(scores, label=class_name)
    
    plt.title('Risk Score Distribution by Class')
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'risk_score_distribution.png'))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(class_metrics).T
    metrics_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'))
    
    # Save full classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(all_true_labels, all_predictions, 
                                     target_names=list(STATES.values())))
    
    # Return aggregated metrics
    return {
        'overall_accuracy': overall_accuracy,
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'classification_report': report,
        'sincere_accuracy': sincere_results['accuracy'],
        'cheating_accuracy': cheating_results['accuracy']
    }

def process_test_data(system, data_path, expected_label, verbose=False):
    """
    Process a test dataset and collect predictions
    
    Args:
        system: Trained ProctoringSystem
        data_path: Path to test data
        expected_label: Expected class label
        verbose: Whether to display detailed output
    
    Returns:
        Dictionary with predictions, true labels, risk scores and accuracy
    """
    # Load the data
    if verbose:
        print(f"Processing test data from {data_path}...")
    
    keystroke_data = load_keystroke_data(data_path)
    if verbose:
        print(f"Loaded {len(keystroke_data)} keystroke events")
    
    # Group by user
    user_data = extract_user_data(keystroke_data)
    if verbose:
        print(f"Processing data for {len(user_data)} users...")
    
    # Process each user's data
    predictions = []
    risk_scores = []
    
    for user_id, user_keystrokes in user_data.items():
        # Set user ID in system
        system.set_user_id(user_id)
        
        # Reset keystrokes processor for each user
        system.keystroke_processor = system.keystroke_processor.__class__()
        
        if verbose:
            print(f"Processing {len(user_keystrokes)} keystrokes for user {user_id}")
        
        # Process keystroke events
        for i, keystroke in enumerate(user_keystrokes):
            state, risk_score = system.process_keystroke(keystroke)
            predictions.append(state)
            risk_scores.append(risk_score)
            
            # Print progress occasionally
            if verbose and i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(user_keystrokes)} keystrokes "
                     f"({i/len(user_keystrokes)*100:.1f}%)")
    
    # Create true labels (all expected_label)
    true_labels = [expected_label] * len(predictions)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'risk_scores': risk_scores,
        'accuracy': accuracy
    }

def print_summary(metrics, using_optimized):
    """Print a summary of the evaluation results"""
    print("\n" + "="*80)
    print(f"MODEL EVALUATION SUMMARY {'(OPTIMIZED VERSION)' if using_optimized else ''}")
    print("="*80)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Sincere Data Accuracy: {metrics['sincere_accuracy']:.4f}")
    print(f"Cheating Data Accuracy: {metrics['cheating_accuracy']:.4f}")
    
    print("\nPer-Class Metrics:")
    for state, state_metrics in metrics['class_metrics'].items():
        print(f"  {state}:")
        print(f"    Precision: {state_metrics['precision']:.4f}")
        print(f"    Recall:    {state_metrics['recall']:.4f}")
        print(f"    F1 Score:  {state_metrics['f1_score']:.4f}")
        print(f"    Support:   {state_metrics['support']}")
    
    print("\nConfusion Matrix:")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {STATES[i]}: {row}")
    
    print("\nDetailed metrics and visualizations have been saved to the output directory.")
    print("="*80)

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    start_time = datetime.now()
    system = train_model(args.sincere_train, args.cheating_train, args.verbose)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate the model
    start_time = datetime.now()
    metrics = evaluate_model(system, args.sincere_test, args.cheating_test, args.output_dir, args.verbose)
    evaluation_time = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    print_summary(metrics, USING_OPTIMIZED)
    
    # Save timing information
    timing_info = {
        'training_time_seconds': training_time,
        'evaluation_time_seconds': evaluation_time,
        'using_optimized': USING_OPTIMIZED
    }
    
    with open(os.path.join(args.output_dir, 'timing_info.json'), 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    # Return the overall accuracy for the bash script
    print(f"OVERALL_ACCURACY:{metrics['overall_accuracy']}")
    print(f"SINCERE_ACCURACY:{metrics['sincere_accuracy']}")
    print(f"CHEATING_ACCURACY:{metrics['cheating_accuracy']}")

if __name__ == "__main__":
    main()
