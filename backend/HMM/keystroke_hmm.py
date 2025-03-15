import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Define the states for the proctoring system
STATES = {
    0: "Not Cheating",
    1: "Distracted or Thinking",
    2: "Maybe Cheating",
    3: "Definitely Cheating"
}

class KeystrokeProcessor:
    """
    Process keystroke data and maintain sliding windows
    """
    def __init__(self, primary_window_size=45, baseline_window_size=240):
        """
        Initialize the keystroke processor
        
        Args:
            primary_window_size: Size of primary window in seconds (default: 45)
            baseline_window_size: Size of baseline window in seconds (default: 240)
        """
        self.primary_window_size = primary_window_size  # 45 seconds
        self.baseline_window_size = baseline_window_size  # 4 minutes
        
        # Sliding windows
        self.primary_window = []
        self.baseline_window = []
        self.special_keys_buffer = []
        
        # Special key combinations to monitor
        self.special_keys = ["Tab", "Alt", "Ctrl", "Meta", "Win"]
        self.special_combinations = [
            ("Ctrl", "Tab"), ("Alt", "Tab"), ("Ctrl", "C"), 
            ("Ctrl", "V"), ("Alt", "Tab"), ("Win", "Tab")
        ]
        
        # To track key down events for calculating hold duration
        self.key_down_tracker = {}
    
    def add_keystroke(self, keystroke):
        """
        Add a new keystroke to the sliding windows
        
        Args:
            keystroke: Dictionary containing keystroke data (timestamp, key, event_type)
        """
        timestamp = keystroke['timestamp']
        key = keystroke['key']
        event_type = keystroke['event_type']
        
        # Update tracking for key down/up events
        if event_type == 'KD':
            self.key_down_tracker[key] = timestamp
        elif event_type == 'KU' and key in self.key_down_tracker:
            # Calculate hold duration
            hold_duration = (timestamp - self.key_down_tracker[key]).total_seconds()
            keystroke['hold_duration'] = hold_duration
            del self.key_down_tracker[key]
        
        # Update primary window (45 seconds)
        self.primary_window.append(keystroke)
        
        # Update baseline window (4 minutes)
        self.baseline_window.append(keystroke)
        
        # Track special keys
        if key in self.special_keys or any(combo[0] == key or combo[1] == key for combo in self.special_combinations):
            self.special_keys_buffer.append(keystroke)
        
        # Clean up windows based on time threshold
        self._clean_windows(timestamp)
    
    def _clean_windows(self, current_time):
        """
        Remove keystrokes older than the window size
        
        Args:
            current_time: Current timestamp
        """
        primary_threshold = current_time - timedelta(seconds=self.primary_window_size)
        self.primary_window = [k for k in self.primary_window if k['timestamp'] >= primary_threshold]
        
        baseline_threshold = current_time - timedelta(seconds=self.baseline_window_size)
        self.baseline_window = [k for k in self.baseline_window if k['timestamp'] >= baseline_threshold]
        
        # Keep special keys buffer for the baseline window timeframe
        self.special_keys_buffer = [k for k in self.special_keys_buffer if k['timestamp'] >= baseline_threshold]
    
    def get_features(self):
        """
        Extract features from current windows
        
        Returns:
            Dictionary of features
        """
        if len(self.primary_window) < 2 or len(self.baseline_window) < 2:
            return None
        
        # Calculate inter-key intervals (IKI)
        primary_ikis = self._calculate_ikis(self.primary_window)
        baseline_ikis = self._calculate_ikis(self.baseline_window)
        
        # Calculate hold durations
        primary_hold_durations = self._calculate_hold_durations(self.primary_window)
        baseline_hold_durations = self._calculate_hold_durations(self.baseline_window)
        
        # Calculate special keys frequency
        special_keys_freq = self._calculate_special_keys_frequency()
        
        # Calculate similarity metrics
        dtw_distance = self._calculate_dtw_distance(primary_ikis, baseline_ikis)
        
        # Calculate statistical features
        primary_iki_mean = np.mean(primary_ikis) if primary_ikis else 0
        primary_iki_std = np.std(primary_ikis) if primary_ikis else 0
        baseline_iki_mean = np.mean(baseline_ikis) if baseline_ikis else 0
        baseline_iki_std = np.std(baseline_ikis) if baseline_ikis else 0
        
        primary_hold_mean = np.mean(primary_hold_durations) if primary_hold_durations else 0
        primary_hold_std = np.std(primary_hold_durations) if primary_hold_durations else 0
        baseline_hold_mean = np.mean(baseline_hold_durations) if baseline_hold_durations else 0
        baseline_hold_std = np.std(baseline_hold_durations) if baseline_hold_durations else 0
        
        # Calculate typing speed
        primary_typing_speed = len(self.primary_window) / self.primary_window_size if self.primary_window else 0
        baseline_typing_speed = len(self.baseline_window) / self.baseline_window_size if self.baseline_window else 0
        
        # Deviation from baseline
        iki_mean_deviation = abs(primary_iki_mean - baseline_iki_mean) / (baseline_iki_mean if baseline_iki_mean else 1)
        hold_mean_deviation = abs(primary_hold_mean - baseline_hold_mean) / (baseline_hold_mean if baseline_hold_mean else 1)
        typing_speed_deviation = abs(primary_typing_speed - baseline_typing_speed) / (baseline_typing_speed if baseline_typing_speed else 1)
        
        return {
            'iki_mean': primary_iki_mean,
            'iki_std': primary_iki_std,
            'hold_mean': primary_hold_mean,
            'hold_std': primary_hold_std,
            'typing_speed': primary_typing_speed,
            'special_keys_freq': special_keys_freq,
            'dtw_distance': dtw_distance,
            'iki_mean_deviation': iki_mean_deviation,
            'hold_mean_deviation': hold_mean_deviation,
            'typing_speed_deviation': typing_speed_deviation
        }
    
    def _calculate_ikis(self, keystrokes):
        """
        Calculate inter-key intervals from keystrokes
        
        Args:
            keystrokes: List of keystroke events
            
        Returns:
            List of inter-key intervals in seconds
        """
        # Filter to only key down events to calculate IKI
        key_down_events = [k for k in keystrokes if k['event_type'] == 'KD']
        key_down_events.sort(key=lambda x: x['timestamp'])
        
        ikis = []
        for i in range(1, len(key_down_events)):
            iki = (key_down_events[i]['timestamp'] - key_down_events[i-1]['timestamp']).total_seconds()
            # Filter out unreasonable values (too long pauses)
            if iki < 5:  # Threshold of 5 seconds
                ikis.append(iki)
                
        return ikis
    
    def _calculate_hold_durations(self, keystrokes):
        """
        Calculate key hold durations
        
        Args:
            keystrokes: List of keystroke events
            
        Returns:
            List of hold durations in seconds
        """
        hold_durations = []
        for k in keystrokes:
            if 'hold_duration' in k:
                # Filter out unreasonable values
                if k['hold_duration'] < 2:  # Threshold of 2 seconds
                    hold_durations.append(k['hold_duration'])
                    
        return hold_durations
    
    def _calculate_special_keys_frequency(self):
        """
        Calculate frequency of special keys usage
        
        Returns:
            Frequency of special keys usage (count per second)
        """
        if not self.special_keys_buffer:
            return 0
        
        time_span = (self.special_keys_buffer[-1]['timestamp'] - self.special_keys_buffer[0]['timestamp']).total_seconds()
        if time_span <= 0:
            return 0
            
        return len(self.special_keys_buffer) / time_span
    
    def _calculate_dtw_distance(self, primary_sequence, baseline_sequence):
        """
        Calculate DTW distance between primary and baseline sequences
        
        Args:
            primary_sequence: List of values from primary window
            baseline_sequence: List of values from baseline window
            
        Returns:
            DTW distance
        """
        if not primary_sequence or not baseline_sequence:
            return 0
            
        # If sequences are too long, sample them
        max_samples = 100
        if len(primary_sequence) > max_samples:
            primary_indices = np.linspace(0, len(primary_sequence)-1, max_samples, dtype=int)
            primary_sequence = [primary_sequence[i] for i in primary_indices]
            
        if len(baseline_sequence) > max_samples:
            baseline_indices = np.linspace(0, len(baseline_sequence)-1, max_samples, dtype=int)
            baseline_sequence = [baseline_sequence[i] for i in baseline_indices]
        
        # Calculate DTW distance
        try:
            distance, _ = fastdtw(np.array(primary_sequence).reshape(-1, 1), 
                                 np.array(baseline_sequence).reshape(-1, 1), 
                                 dist=euclidean)
            return distance
        except:
            return 0


class ProctoringHMM:
    """
    Hidden Markov Model for proctoring system
    """
    def __init__(self, n_components=4):
        """
        Initialize the HMM model
        
        Args:
            n_components: Number of hidden states (default: 4)
        """
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, 
                                covariance_type="full", 
                                n_iter=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Store reference patterns for different states
        self.reference_patterns = {
            0: [],  # Not Cheating
            1: [],  # Distracted or Thinking
            2: [],  # Maybe Cheating 
            3: []   # Definitely Cheating
        }
        
    def train(self, features_list, labels=None):
        """
        Train the HMM model
        
        Args:
            features_list: List of feature dictionaries
            labels: List of state labels (optional), if provided will be used for supervised training
            
        Returns:
            True if training was successful, False otherwise
        """
        if not features_list:
            return False
            
        # Convert list of dictionaries to numpy array
        feature_names = list(features_list[0].keys())
        data = np.array([[feat[name] for name in feature_names] for feat in features_list])
        
        # Scale the features
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        
        # If labels are provided, use them to create reference patterns
        if labels is not None and len(labels) == len(features_list):
            for i, label in enumerate(labels):
                if label in self.reference_patterns:
                    self.reference_patterns[label].append(scaled_data[i])
        
        # Reshape for HMM (n_samples, n_features)
        try:
            self.model.fit(scaled_data)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to train HMM: {e}")
            return False
    
    def predict_state(self, features):
        """
        Predict the current state based on features
        
        Args:
            features: Dictionary of features
            
        Returns:
            Predicted state, risk score
        """
        if not self.is_trained or not features:
            return 0, 0.0
            
        # Convert features to numpy array
        feature_names = list(features.keys())
        data = np.array([[features[name] for name in feature_names]])
        
        # Scale the features
        scaled_data = self.scaler.transform(data)
        
        # Predict the most likely state
        try:
            state = self.model.predict(scaled_data)[0]
            
            # Calculate risk score based on state probabilities
            log_probs = self.model.score_samples(scaled_data)
            risk_score = 1.0 - np.exp(log_probs[0]) / 4.0  # Normalize
            risk_score = max(0.0, min(1.0, risk_score))  # Clamp to [0, 1]
            
            # Calculate similarity to reference patterns if available
            if sum(len(patterns) for patterns in self.reference_patterns.values()) > 0:
                # Find the closest reference pattern and adjust score
                min_distances = {}
                for state_label, patterns in self.reference_patterns.items():
                    if patterns:
                        distances = [euclidean(scaled_data[0], pattern) for pattern in patterns]
                        min_distances[state_label] = min(distances) if distances else float('inf')
                
                # If we have reference patterns for multiple states
                if len(min_distances) > 1:
                    # Determine the closest state
                    closest_state = min(min_distances, key=min_distances.get)
                    second_closest = sorted(min_distances.items(), key=lambda x: x[1])[1][0]
                    
                    # Calculate distance ratio between closest and second closest
                    ratio = min_distances[closest_state] / (min_distances[second_closest] + 1e-10)
                    
                    # Adjust state and risk score based on reference pattern match
                    if ratio < 0.7:  # Strong match to reference pattern
                        state = closest_state
                        
                        # Adjust risk score based on the state
                        if closest_state == 0:  # Not Cheating
                            risk_score = min(0.3, risk_score)
                        elif closest_state == 3:  # Definitely Cheating
                            risk_score = max(0.8, risk_score)
            
            return state, risk_score
        except Exception as e:
            print(f"Error predicting state: {e}")
            return 0, 0.0


class ProctoringSystem:
    """
    Main proctoring system that integrates keystroke processing and HMM classification
    """
    def __init__(self, pretrained=False):
        """
        Initialize the proctoring system
        
        Args:
            pretrained: Whether to use a pretrained model (default: False)
        """
        self.keystroke_processor = KeystrokeProcessor()
        self.hmm = ProctoringHMM()
        self.feature_history = []
        self.state_history = []
        self.risk_score_history = []
        self.pretrained = pretrained
        self.user_id = None
        
    def set_user_id(self, user_id):
        """
        Set the current user ID
        
        Args:
            user_id: User ID string
        """
        self.user_id = user_id
        
    def process_keystroke(self, keystroke):
        """
        Process a new keystroke event
        
        Args:
            keystroke: Dictionary containing keystroke data
        
        Returns:
            Current state and risk score
        """
        self.keystroke_processor.add_keystroke(keystroke)
        
        # Extract features
        features = self.keystroke_processor.get_features()
        if not features:
            return 0, 0.0
            
        self.feature_history.append(features)
        
        # For the first few keystrokes, build the baseline
        if len(self.feature_history) < 10 and not self.pretrained:
            return 0, 0.0
            
        # Train or update the model periodically
        if (len(self.feature_history) % 20 == 0 or not self.hmm.is_trained) and not self.pretrained:
            self.hmm.train(self.feature_history)
        
        # Predict the current state
        state, risk_score = self.hmm.predict_state(features)
        
        # Apply rule-based adjustments
        state, risk_score = self._apply_rules(features, state, risk_score)
        
        # Store history
        self.state_history.append(state)
        self.risk_score_history.append(risk_score)
        
        return state, risk_score
    
    def _apply_rules(self, features, state, risk_score):
        """
        Apply additional rule-based adjustments to the state and risk score
        
        Args:
            features: Dictionary of features
            state: Current state from HMM
            risk_score: Current risk score
            
        Returns:
            Adjusted state and risk score
        """
        # Check for high special keys frequency
        if features['special_keys_freq'] > 0.5:  # More than 0.5 special keys per second
            risk_score = min(1.0, risk_score + 0.2)
            
        # Check for high deviation from baseline
        if features['iki_mean_deviation'] > 0.5 or features['hold_mean_deviation'] > 0.5:
            risk_score = min(1.0, risk_score + 0.15)
            
        # Check for high DTW distance
        if features['dtw_distance'] > 100:
            risk_score = min(1.0, risk_score + 0.1)
            
        # Determine state based on adjusted risk score
        if risk_score < 0.3:
            state = 0  # Not Cheating
        elif risk_score < 0.5:
            state = 1  # Distracted or Thinking
        elif risk_score < 0.8:
            state = 2  # Maybe Cheating
        else:
            state = 3  # Definitely Cheating
            
        return state, risk_score
    
    def get_status(self):
        """
        Get the current status of the proctoring system
        
        Returns:
            Dictionary containing current state and risk score
        """
        if not self.state_history:
            return {"state": "Not Cheating", "state_code": 0, "risk_score": 0.0}
            
        current_state = self.state_history[-1]
        current_risk = self.risk_score_history[-1]
        
        return {
            "state": STATES[current_state],
            "state_code": current_state,
            "risk_score": current_risk
        }
    
    def pretrain_with_datasets(self, sincere_path, cheating_path, sample_size=1000):
        """
        Pretrain the HMM model with labeled datasets
        
        Args:
            sincere_path: Path to sincere (not cheating) data
            cheating_path: Path to cheating data
            sample_size: Number of samples to use from each dataset (default: 1000)
            
        Returns:
            True if pretraining was successful, False otherwise
        """
        print(f"Pretraining HMM model with datasets: {sincere_path} and {cheating_path}")
        
        # Load sincere data
        sincere_data = load_keystroke_data(sincere_path)
        if len(sincere_data) > sample_size:
            sincere_data = sincere_data[:sample_size]
        
        # Load cheating data
        cheating_data = load_keystroke_data(cheating_path)
        if len(cheating_data) > sample_size:
            cheating_data = cheating_data[:sample_size]
        
        print(f"Loaded {len(sincere_data)} sincere events and {len(cheating_data)} cheating events")
        
        # Process sincere data
        sincere_features = []
        for keystroke in sincere_data:
            self.keystroke_processor.add_keystroke(keystroke)
            features = self.keystroke_processor.get_features()
            if features:
                sincere_features.append(features)
            
            # Reset processor after a window of events to handle multiple users
            if len(sincere_features) % 100 == 0:
                self.keystroke_processor = KeystrokeProcessor()
        
        # Reset processor for cheating data
        self.keystroke_processor = KeystrokeProcessor()
        
        # Process cheating data
        cheating_features = []
        for keystroke in cheating_data:
            self.keystroke_processor.add_keystroke(keystroke)
            features = self.keystroke_processor.get_features()
            if features:
                cheating_features.append(features)
                
            # Reset processor after a window of events to handle multiple users
            if len(cheating_features) % 100 == 0:
                self.keystroke_processor = KeystrokeProcessor()
        
        # Reset processor for real-time use
        self.keystroke_processor = KeystrokeProcessor()
        
        # Combine features and create labels
        all_features = sincere_features + cheating_features
        labels = [0] * len(sincere_features) + [3] * len(cheating_features)
        
        if not all_features:
            print("Failed to extract features for pretraining")
            return False
        
        # Train the HMM model
        success = self.hmm.train(all_features, labels)
        
        if success:
            self.pretrained = True
            print(f"Successfully pretrained HMM model with {len(all_features)} feature sets")
        else:
            print("Failed to pretrain HMM model")
        
        return success


def load_keystroke_data(json_path):
    """
    Load keystroke data from a JSON file
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of keystroke events
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    keystroke_data = []

    for user_id, user_data in data.items():  # Iterate over user IDs
        if "keyboard_data" in user_data:  # Check if keyboard data exists
            for event in user_data["keyboard_data"]:
                keystroke_data.append({
                    'timestamp': datetime.fromtimestamp(event[2] / 1000),  # Convert from milliseconds
                    'key': event[1],
                    'event_type': event[0],
                    'user_id': user_id  # Add user_id to track different users
                })
    
    # Sort by timestamp
    keystroke_data.sort(key=lambda x: x['timestamp'])
    
    return keystroke_data


def extract_user_data(keystroke_data):
    """
    Extract keystroke data by user
    
    Args:
        keystroke_data: List of keystroke events
        
    Returns:
        Dictionary mapping user IDs to their keystroke data
    """
    user_data = defaultdict(list)
    
    for keystroke in keystroke_data:
        user_id = keystroke.get('user_id', 'unknown')
        user_data[user_id].append(keystroke)
    
    return user_data 