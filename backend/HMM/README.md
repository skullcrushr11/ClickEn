# Keystroke Dynamics Proctoring System

A proctoring system that uses keystroke dynamics and mouse movement data to classify user behavior into four states using a continuous Hidden Markov Model (HMM). The system supports multi-user training and can differentiate between normal and cheating behavior patterns.

## Features

- Real-time analysis of keystroke patterns
- Classification into four user states:
  - Not Cheating
  - Distracted or Thinking
  - Maybe Cheating
  - Definitely Cheating
- Dynamic risk score calculation
- Multi-user training and personalization
- Visualization of user behavior over time
- Evaluation tools for system performance analysis
- Cross-validation by user for robust model evaluation

## Requirements

- Python 3.8+
- Required libraries listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo

The demo script processes keystroke data and displays real-time classification. It can now be run with pretraining on both sincere and cheating datasets:

```
python demo.py --data sincere.json --sincere sincere.json --cheating cheating.json --speed 10
```

Arguments:
- `--data`: Path to the keystroke data JSON file to analyze (default: sincere.json)
- `--sincere`: Path to sincere (not cheating) data for pretraining (default: sincere.json)
- `--cheating`: Path to cheating data for pretraining (default: cheating.json)
- `--speed`: Playback speed multiplier (default: 1.0)
- `--no-plot`: Disable real-time plotting
- `--no-pretrain`: Disable pretraining with labeled data
- `--user-id`: Specific user ID to monitor (default: None, uses all users)

### Evaluating the System

The evaluation script processes both sincere and cheating data, and generates performance metrics:

```
python evaluate.py --sincere sincere.json --cheating cheating.json --output evaluation_results
```

Arguments:
- `--sincere`: Path to sincere keystroke data (default: sincere.json)
- `--cheating`: Path to cheating keystroke data (default: cheating.json)
- `--output`: Output directory for results (default: evaluation_results)
- `--no-pretrain`: Disable pretraining with labeled data
- `--user-ids`: Specific user IDs to evaluate as comma-separated list (default: None, evaluate all)
- `--cross-validate`: Perform cross-validation by user (default: False)

#### Cross-validation Mode

The system supports cross-validation by user to better evaluate model generalizability:

```
python evaluate.py --sincere sincere.json --cheating cheating.json --output cross_val_results --cross-validate
```

This mode trains the model on all users except one and tests on the excluded user, repeating for all users in both datasets.

## How It Works

The system uses the following components:

1. **Keystroke Processor**: Maintains sliding windows of keystroke data and extracts features.
2. **Continuous HMM**: Models user behavior patterns and classifies them into states.
3. **Feature Extraction**: Calculates metrics like inter-key intervals, hold durations, typing speed, etc.
4. **Similarity Measurement**: Uses Dynamic Time Warping (DTW) to compare current behavior with baseline.
5. **Risk Score Calculation**: Combines multiple factors to generate a dynamic risk score.
6. **Multi-user Support**: Handles data from multiple users for more robust training.

### Key Features Analyzed

- Inter-key time intervals (IKI)
- Key hold durations
- Typing speed
- Special key usage (Ctrl, Alt, Tab, etc.)
- Similarity between recent and baseline behavior
- User-specific typing patterns

### Data Format

The system works with JSON files containing keystroke data from multiple users. Each user's data includes:
- Key press events with timestamps
- Key release events with timestamps
- Special key combinations

## Improvements with Multi-user Training

The system now includes several improvements for multi-user training:

1. **Reference Pattern Storage**: The HMM model stores reference patterns for each state from the training data
2. **Per-user Pattern Analysis**: Processes each user's data separately when training
3. **Cross-user Generalization**: Learns patterns that generalize across different users
4. **Similarity-based Classification**: Uses similarity to reference patterns to improve state classification
5. **Cross-validation Support**: Evaluates model performance across different users

## System Architecture

```
┌───────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Keystroke         │ -> │ Feature          │ -> │ HMM             │
│ Processor         │    │ Extraction       │    │ Classification   │
└───────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌───────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Sliding Windows   │    │ Similarity       │    │ Risk Score      │
│ Management        │    │ Calculation      │    │ Generation      │
└───────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 v
                        ┌─────────────────┐
                        │ User Pattern    │
                        │ Database        │
                        └─────────────────┘
```

## Output

The system generates:
- Real-time classification and risk scores
- CSV files with detailed feature data
- Visualizations of risk scores and state classifications
- Performance metrics and evaluations
- Cross-validation reports by user

## License

This project is released under the MIT License. 