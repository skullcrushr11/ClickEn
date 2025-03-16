#!/bin/bash

# Simple script to run the simplified keystroke HMM

# Check if input files are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <sincere_data.json> <cheating_data.json>"
    exit 1
fi

SINCERE_DATA=$1
CHEATING_DATA=$2
OUTPUT_DIR="simple_hmm_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the simple HMM implementation
echo "Running simple keystroke HMM..."
python simple_keystroke_hmm.py --sincere $SINCERE_DATA --cheating $CHEATING_DATA --output-dir $OUTPUT_DIR

echo "Done! Results saved to $OUTPUT_DIR" 