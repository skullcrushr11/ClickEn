import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import random
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
        
        # Sliding windows - using deque for better performance
        self.primary_window = deque(maxlen=1000)  # Reasonable upper limit
        self.baseline_window = deque(maxlen=5000)  # Reasonable upper limit
        self.special_keys_buffer = deque(maxlen=1000)  # Reasonable upper limit
        
        # Special key combinations to monitor
        self.special_keys = frozenset(["Tab", "Alt", "Ctrl", "Meta", "Win"])  # Use frozenset for faster lookups
        self.special_combinations = [
            ("Ctrl", "Tab"), ("Alt", "Tab"), ("Ctrl", "C"), 
            ("Ctrl", "V"), ("Alt", "Tab"), ("Win", "Tab")
        ]
        
        # To track key down events for calculating hold duration
        self.key_down_tracker = {}
        
        # Cache the primary and baseline thresholds
        self.primary_threshold = None
        self.baseline_threshold = None
        self.last_update_time = None
    
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
        
        # Only clean windows every second to reduce overhead
        if self.last_update_time is None or (timestamp - self.last_update_time).total_seconds() >= 1.0:
            self._clean_windows(timestamp)
            self.last_update_time = timestamp
    
    def _clean_windows(self, current_time):
        """
        Remove keystrokes older than the window size
        
        Args:
            current_time: Current timestamp
        """
        primary_threshold = current_time - timedelta(seconds=self.primary_window_size)
        baseline_threshold = current_time - timedelta(seconds=self.baseline_window_size)
        
        # Store thresholds for faster filtering
        self.primary_threshold = primary_threshold
        self.baseline_threshold = baseline_threshold
        
        # Use list comprehension for faster filtering
        self.primary_window = deque([k for k in self.primary_window if k['timestamp'] >= primary_threshold], 
                                   maxlen=self.primary_window.maxlen)
        
        self.baseline_window = deque([k for k in self.baseline_window if k['timestamp'] >= baseline_threshold], 
                                    maxlen=self.baseline_window.maxlen)
        
        # Keep special keys buffer for the baseline window timeframe
        self.special_keys_buffer = deque([k for k in self.special_keys_buffer if k['timestamp'] >= baseline_threshold], 
                                        maxlen=self.special_keys_buffer.maxlen)
    
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
        
        # Calculate statistical features - use numpy for performance
        primary_ikis_np = np.array(primary_ikis) if primary_ikis else np.array([0])
        baseline_ikis_np = np.array(baseline_ikis) if baseline_ikis else np.array([0])
        primary_hold_np = np.array(primary_hold_durations) if primary_hold_durations else np.array([0])
        baseline_hold_np = np.array(baseline_hold_durations) if baseline_hold_durations else np.array([0])
        
        primary_iki_mean = np.mean(primary_ikis_np)
        primary_iki_std = np.std(primary_ikis_np)
        baseline_iki_mean = np.mean(baseline_ikis_np)
        baseline_iki_std = np.std(baseline_ikis_np)
        
        primary_hold_mean = np.mean(primary_hold_np)
        primary_hold_std = np.std(primary_hold_np)
        baseline_hold_mean = np.mean(baseline_hold_np)
        baseline_hold_std = np.std(baseline_hold_np)
        
        # Calculate typing speed
        primary_typing_speed = len(self.primary_window) / self.primary_window_size if self.primary_window else 0
        baseline_typing_speed = len(self.baseline_window) / self.baseline_window_size if self.baseline_window else 0
        
        # Deviation from baseline - with safety checks for zero division
        iki_mean_deviation = abs(primary_iki_mean - baseline_iki_mean) / max(baseline_iki_mean, 1e-10)
        hold_mean_deviation = abs(primary_hold_mean - baseline_hold_mean) / max(baseline_hold_mean, 1e-10)
        typing_speed_deviation = abs(primary_typing_speed - baseline_typing_speed) / max(baseline_typing_speed, 1e-10)
        
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
        
        # Sort if needed (might be already sorted)
        if len(key_down_events) > 1 and key_down_events[0]['timestamp'] > key_down_events[-1]['timestamp']:
            key_down_events.sort(key=lambda x: x['timestamp'])
        
        # Pre-allocate result for better performance
        ikis = []
        ikis_append = ikis.append  # Local function reference for speed
        
        # Use a threshold of 5 seconds for unreasonable values
        for i in range(1, len(key_down_events)):
            iki = (key_down_events[i]['timestamp'] - key_down_events[i-1]['timestamp']).total_seconds()
            if iki < 5:  # Threshold for unreasonable values
                ikis_append(iki)
                
        return ikis
    
    def _calculate_hold_durations(self, keystrokes):
        """
        Calculate key hold durations
        
        Args:
            keystrokes: List of keystroke events
            
        Returns:
            List of hold durations in seconds
        """
        # Pre-allocate result for better performance
        hold_durations = []
        hold_durations_append = hold_durations.append  # Local function reference for speed
        
        for k in keystrokes:
            if 'hold_duration' in k and k['hold_duration'] < 2:  # Filter unreasonable values in one step
                hold_durations_append(k['hold_duration'])
                
        return hold_durations
    
    def _calculate_special_keys_frequency(self):
        """
        Calculate frequency of special keys usage
        
        Returns:
            Frequency of special keys usage (count per second)
        """
        n_keys = len(self.special_keys_buffer)
        if n_keys < 2:
            return 0
        
        time_span = (self.special_keys_buffer[-1]['timestamp'] - self.special_keys_buffer[0]['timestamp']).total_seconds()
        if time_span <= 0:
            return 0
            
        return n_keys / time_span
    
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
        max_samples = 50  # Reduced from 100 for speed
        if len(primary_sequence) > max_samples:
            # Use linear indices for faster sampling
            indices = np.linspace(0, len(primary_sequence)-1, max_samples, dtype=int)
            primary_sequence = [primary_sequence[i] for i in indices]
            
        if len(baseline_sequence) > max_samples:
            indices = np.linspace(0, len(baseline_sequence)-1, max_samples, dtype=int)
            baseline_sequence = [baseline_sequence[i] for i in indices]
        
        # Calculate DTW distance - use try/except for safety
        try:
            # Convert to numpy arrays once, outside the fastdtw call
            primary_np = np.array(primary_sequence).reshape(-1, 1)
            baseline_np = np.array(baseline_sequence).reshape(-1, 1)
            distance, _ = fastdtw(primary_np, baseline_np, dist=euclidean)
            return distance
        except Exception:
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
                                covariance_type="diag",  # Changed from "full" to "diag" for better convergence
                                n_iter=100,
                                tol=1e-3,               # Increased tolerance for better convergence
                                algorithm="viterbi")    # Specify algorithm
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Store reference patterns for different states
        self.reference_patterns = {
            0: [],  # Not Cheating
            1: [],  # Distracted or Thinking
            2: [],  # Maybe Cheating 
            3: []   # Definitely Cheating
        }
        
        # Cache for euclidean distance calculations
        self._distance_cache = {}
        
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
        
        # Clear distance cache on retraining
        self._distance_cache.clear()
            
        # Convert list of dictionaries to numpy array - optimize by building all at once
        if features_list:
            feature_names = list(features_list[0].keys())
            data = np.array([[feat.get(name, 0) for name in feature_names] for feat in features_list])
        else:
            return False
        
        # Apply outlier removal using IQR
        try:
            # Calculate IQR for each feature
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            
            # Define bounds for outlier detection
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Create mask for inliers
            mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
            
            # If we have too few samples after filtering, keep original data
            if np.sum(mask) < len(data) * 0.7:
                print(f"Outlier removal would discard too many samples ({len(data) - np.sum(mask)} out of {len(data)}). Using original data.")
                filtered_data = data
                filtered_labels = labels
            else:
                # Filter the data and labels
                filtered_data = data[mask]
                filtered_labels = [labels[i] for i in range(len(labels)) if mask[i]] if labels is not None else None
                print(f"Removed {len(data) - len(filtered_data)} outlier samples out of {len(data)} total samples.")
            
            # Update data and labels
            data = filtered_data
            labels = filtered_labels
        except Exception as e:
            print(f"Error during outlier removal: {e}. Using original data.")
        
        # Scale the features
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        
        # If labels are provided, use them to create reference patterns
        if labels is not None and len(labels) == len(features_list):
            # Clear existing patterns
            for state in self.reference_patterns:
                self.reference_patterns[state] = []
                
            # Add new patterns
            for i, label in enumerate(labels):
                if i < len(scaled_data) and label in self.reference_patterns:
                    self.reference_patterns[label].append(scaled_data[i])
        
        # Try training with progressively simpler models if needed
        covariance_types = ["diag", "spherical", "tied"]
        success = False
        
        for cov_type in covariance_types:
            try:
                print(f"Attempting training with covariance_type={cov_type}...")
                # Create a new model with current settings
                self.model = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=cov_type,
                    n_iter=100,
                    tol=1e-3,
                    algorithm="viterbi",
                    random_state=42  # Set random state for reproducibility
                )
                
                # Fit the model
                self.model.fit(scaled_data)
                success = True
                print(f"Successfully trained HMM with covariance_type={cov_type}")
                break  # Exit loop if successful
            except Exception as e:
                print(f"Failed to train with covariance_type={cov_type}: {e}")
                
                # If we failed with the current covariance type, try with a smaller dataset
                if len(scaled_data) > 500:
                    try:
                        reduced_size = min(500, int(len(scaled_data) * 0.5))
                        print(f"Trying with reduced dataset of {reduced_size} samples...")
                        indices = np.random.choice(len(scaled_data), reduced_size, replace=False)
                        reduced_data = scaled_data[indices]
                        
                        self.model = GaussianHMM(
                            n_components=self.n_components,
                            covariance_type=cov_type,
                            n_iter=100,
                            tol=1e-3,
                            algorithm="viterbi",
                            random_state=42
                        )
                        self.model.fit(reduced_data)
                        success = True
                        print(f"Successfully trained HMM with covariance_type={cov_type} on reduced dataset")
                        break
                    except Exception as e2:
                        print(f"Failed with reduced dataset: {e2}")
        
        # If we couldn't train with any covariance type, try with default model but fewer components
        if not success and self.n_components > 2:
            try:
                print(f"Attempting training with reduced complexity (2 components)...")
                # Create a simpler model
                self.model = GaussianHMM(
                    n_components=2,  # Reduced complexity
                    covariance_type="diag",
                    n_iter=50,  # Fewer iterations
                    tol=1e-2,  # Even more tolerant
                    algorithm="viterbi",
                    random_state=42
                )
                
                # Fit the model
                self.model.fit(scaled_data)
                self.n_components = 2  # Update our stored value
                success = True
                print(f"Successfully trained simplified HMM model with 2 components")
            except Exception as e:
                print(f"Failed to train simplified model: {e}")
        
        self.is_trained = success
        return success
    
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
        data = np.array([[features.get(name, 0) for name in feature_names]])
        
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
            total_patterns = sum(len(patterns) for patterns in self.reference_patterns.values())
            if total_patterns > 0:
                # Find the closest reference pattern and adjust score
                min_distances = {}
                
                # Generate a hash for the current feature vector for caching
                feature_tuple = tuple(scaled_data[0])
                
                for state_label, patterns in self.reference_patterns.items():
                    if patterns:
                        # Find minimum distance for each state using cache where possible
                        distances = []
                        for pattern in patterns:
                            pattern_tuple = tuple(pattern)
                            cache_key = (feature_tuple, pattern_tuple)
                            
                            if cache_key in self._distance_cache:
                                distance = self._distance_cache[cache_key]
                            else:
                                distance = euclidean(scaled_data[0], pattern)
                                # Only cache if not too many items to avoid memory issues
                                if len(self._distance_cache) < 10000:
                                    self._distance_cache[cache_key] = distance
                                    
                            distances.append(distance)
                        
                        min_distances[state_label] = min(distances)
                
                # If we have reference patterns for multiple states
                if len(min_distances) > 1:
                    # Determine the closest state and second closest
                    sorted_distances = sorted(min_distances.items(), key=lambda x: x[1])
                    closest_state = sorted_distances[0][0]
                    second_closest = sorted_distances[1][0]
                    
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
            #print(f"Error predicting state: {e}")
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
        
        # Cache for last state and score
        self.last_features_hash = None
        self.last_state = 0
        self.last_risk_score = 0.0
        
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
        
        # Generate a simple hash of feature values to check if they've changed significantly
        features_hash = hash(tuple(round(v * 100) / 100 for v in features.values()))
        
        # If features haven't changed much, reuse the last prediction to save computation
        if features_hash == self.last_features_hash and self.last_state is not None:
            return self.last_state, self.last_risk_score
            
        # Update the features hash
        self.last_features_hash = features_hash
            
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
        
        # Cache the results
        self.last_state = state
        self.last_risk_score = risk_score
        
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
    
    def _process_keystroke_batch(self, keystroke_batch):
        """
        Process a batch of keystrokes to extract features (for parallel processing)
        
        Args:
            keystroke_batch: List of keystroke events
            
        Returns:
            List of feature dictionaries
        """
        processor = KeystrokeProcessor()
        features = []
        
        for keystroke in keystroke_batch:
            processor.add_keystroke(keystroke)
            current_features = processor.get_features()
            if current_features:
                features.append(current_features)
        
        return features
        
    def pretrain_with_datasets(self, sincere_path, cheating_path, sample_size=2000):
        """
        Pretrain the HMM model with labeled datasets
        
        Args:
            sincere_path: Path to sincere (not cheating) data
            cheating_path: Path to cheating data
            sample_size: Number of samples to use from each dataset (default: 2000)
            
        Returns:
            True if pretraining was successful, False otherwise
        """
        print(f"Pretraining HMM model with datasets: {sincere_path} and {cheating_path}")
        
        # Load sincere data
        sincere_data = load_keystroke_data(sincere_path)
        if len(sincere_data) > sample_size:
            # Instead of taking the first N samples, take a random sample
            random.seed(42)  # For reproducibility
            sincere_data = random.sample(sincere_data, sample_size)
            print(f"Randomly sampled {sample_size} events from sincere data")
        
        # Load cheating data
        cheating_data = load_keystroke_data(cheating_path)
        if len(cheating_data) > sample_size:
            import random
            random.seed(42)  # For reproducibility
            cheating_data = random.sample(cheating_data, sample_size)
            print(f"Randomly sampled {sample_size} events from cheating data")
        
        print(f"Using {len(sincere_data)} sincere events and {len(cheating_data)} cheating events")
        
        # Process in smaller batches to help with convergence
        batch_size = min(500, sample_size)  # Use smaller batches
        print(f"Processing data in batches of {batch_size}")
        
        # Group data by user for better parallelization
        sincere_users = extract_user_data(sincere_data)
        cheating_users = extract_user_data(cheating_data)
        
        # Use parallel processing for feature extraction if we have a lot of data
        sincere_features = []
        cheating_features = []
        
        # Only use parallel processing if we have a reasonable amount of data
        if len(sincere_data) + len(cheating_data) > 2000 and cpu_count() > 1:
            try:
                # Prepare batches for sincere data
                sincere_batches = []
                for user_keystrokes in sincere_users.values():
                    # Split user data into smaller batches
                    for i in range(0, len(user_keystrokes), batch_size):
                        batch = user_keystrokes[i:i + batch_size]
                        sincere_batches.append(batch)
                
                # Prepare batches for cheating data
                cheating_batches = []
                for user_keystrokes in cheating_users.values():
                    for i in range(0, len(user_keystrokes), batch_size):
                        batch = user_keystrokes[i:i + batch_size]
                        cheating_batches.append(batch)
                
                # Process batches in parallel
                n_workers = min(cpu_count(), 4)  # Limit to 4 cores max for stability
                with Pool(processes=n_workers) as pool:
                    sincere_results = pool.map(self._process_keystroke_batch, sincere_batches)
                    for result in sincere_results:
                        sincere_features.extend(result)
                    
                    cheating_results = pool.map(self._process_keystroke_batch, cheating_batches)
                    for result in cheating_results:
                        cheating_features.extend(result)
                
                print(f"Extracted {len(sincere_features)} sincere features and {len(cheating_features)} cheating features using parallel processing")
                
            except Exception as e:
                print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
                # Reset features in case of failure
                sincere_features = []
                cheating_features = []
        
        # If parallel processing failed or wasn't attempted, use sequential processing
        if not sincere_features or not cheating_features:
            # Process sincere data
            for user_id, user_keystrokes in sincere_users.items():
                processor = KeystrokeProcessor()
                batch_count = 0
                for i in range(0, len(user_keystrokes), batch_size):
                    batch = user_keystrokes[i:min(i + batch_size, len(user_keystrokes))]
                    batch_count += 1
                    
                    for keystroke in batch:
                        processor.add_keystroke(keystroke)
                        features = processor.get_features()
                        if features:
                            sincere_features.append(features)
                    
                    # Reset processor after each batch to avoid memory issues
                    if batch_count % 3 == 0:
                        processor = KeystrokeProcessor()
            
            # Process cheating data
            for user_id, user_keystrokes in cheating_users.items():
                processor = KeystrokeProcessor()
                batch_count = 0
                for i in range(0, len(user_keystrokes), batch_size):
                    batch = user_keystrokes[i:min(i + batch_size, len(user_keystrokes))]
                    batch_count += 1
                    
                    for keystroke in batch:
                        processor.add_keystroke(keystroke)
                        features = processor.get_features()
                        if features:
                            cheating_features.append(features)
                    
                    # Reset processor after each batch to avoid memory issues
                    if batch_count % 3 == 0:
                        processor = KeystrokeProcessor()
            
            print(f"Extracted {len(sincere_features)} sincere features and {len(cheating_features)} cheating features sequentially")
        
        # Reset processor for real-time use
        self.keystroke_processor = KeystrokeProcessor()
        
        # Combine features and create labels
        all_features = sincere_features + cheating_features
        labels = [0] * len(sincere_features) + [3] * len(cheating_features)
        
        if not all_features:
            print("Failed to extract features for pretraining")
            return False
        
        # Balance the dataset if needed
        if len(sincere_features) > len(cheating_features) * 3:
            # If sincere features are much more than cheating features, subsample them
            random.seed(42)
            sincere_indices = random.sample(range(len(sincere_features)), len(cheating_features) * 3)
            sincere_features_balanced = [sincere_features[i] for i in sincere_indices]
            all_features = sincere_features_balanced + cheating_features
            labels = [0] * len(sincere_features_balanced) + [3] * len(cheating_features)
            print(f"Balanced dataset to {len(sincere_features_balanced)} sincere features and {len(cheating_features)} cheating features")
        
        # Make sure features are not empty and contain proper values
        valid_features = []
        valid_labels = []
        for i, features in enumerate(all_features):
            # Check if any feature has NaN or infinity
            has_invalid = any(not np.isfinite(v) for v in features.values())
            if not has_invalid:
                valid_features.append(features)
                valid_labels.append(labels[i])
            
        if len(valid_features) < len(all_features):
            print(f"Removed {len(all_features) - len(valid_features)} features with invalid values")
            all_features = valid_features
            labels = valid_labels
        
        # Try to train with automatic covariance regularization
        try:
            # Train the HMM model with safeguards
            success = self.hmm.train(all_features, labels)
            
            if success:
                self.pretrained = True
                print(f"Successfully pretrained HMM model with {len(all_features)} feature sets")
            else:
                print("Failed to pretrain HMM model - trying with smaller dataset")
                
                # Try with an even smaller dataset
                if len(all_features) > 500:
                    random.seed(42)
                    indices = random.sample(range(len(all_features)), 500)
                    reduced_features = [all_features[i] for i in indices]
                    reduced_labels = [labels[i] for i in indices]
                    
                    print(f"Attempting with reduced dataset of 500 samples")
                    success = self.hmm.train(reduced_features, reduced_labels)
                    
                    if success:
                        self.pretrained = True
                        print(f"Successfully pretrained HMM model with reduced dataset of 500 samples")
        
        except Exception as e:
            print(f"Error during training: {e}")
            success = False
        
        return success


def _parse_json_chunk(chunk):
    """Helper function to parse a chunk of JSON data"""
    result = []
    try:
        data = json.loads(chunk)
        for user_id, user_data in data.items():
            if "keyboard_data" in user_data:
                for event in user_data["keyboard_data"]:
                    result.append({
                        'timestamp': datetime.fromtimestamp(event[2] / 1000),
                        'key': event[1],
                        'event_type': event[0],
                        'user_id': user_id
                    })
    except Exception as e:
        print(f"Error parsing JSON chunk: {e}")
    return result


def load_keystroke_data(json_path):
    """
    Load keystroke data from a JSON file
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of keystroke events
    """
    file_size = os.path.getsize(json_path)
    
    # For small files, use the simple approach
    if file_size < 100 * 1024 * 1024:  # Less than 100MB
        with open(json_path, 'r') as f:
            data = json.load(f)

        keystroke_data = []

        for user_id, user_data in data.items():
            if "keyboard_data" in user_data:
                for event in user_data["keyboard_data"]:
                    keystroke_data.append({
                        'timestamp': datetime.fromtimestamp(event[2] / 1000),
                        'key': event[1],
                        'event_type': event[0],
                        'user_id': user_id
                    })
        
        # Sort by timestamp
        keystroke_data.sort(key=lambda x: x['timestamp'])
        
        return keystroke_data
    
    # For large files, use parallel processing
    try:
        # Try parallel processing for large files
        n_workers = min(cpu_count(), 4)  # Limit to 4 cores max
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        # Read the file in chunks and process in parallel
        keystroke_data = []
        chunks = []
        
        with open(json_path, 'r') as f:
            file_content = f.read()
            
        # We need to ensure we have valid JSON for each chunk
        # For simplicity, we'll just process the whole file
        with Pool(processes=n_workers) as pool:
            # Process the entire file - not ideal but works for this case
            result = pool.apply(_parse_json_chunk, (file_content,))
            keystroke_data.extend(result)
            
        print(f"Loaded {len(keystroke_data)} keystroke events using parallel processing")
        
        # Sort by timestamp
        keystroke_data.sort(key=lambda x: x['timestamp'])
        
        return keystroke_data
        
    except Exception as e:
        print(f"Parallel loading failed: {e}. Falling back to sequential loading.")
        
        # Fall back to sequential loading
        with open(json_path, 'r') as f:
            data = json.load(f)

        keystroke_data = []

        for user_id, user_data in data.items():
            if "keyboard_data" in user_data:
                for event in user_data["keyboard_data"]:
                    keystroke_data.append({
                        'timestamp': datetime.fromtimestamp(event[2] / 1000),
                        'key': event[1],
                        'event_type': event[0],
                        'user_id': user_id
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