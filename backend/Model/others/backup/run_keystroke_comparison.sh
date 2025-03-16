#!/bin/bash

# Script to run the keystroke pattern comparison model

# Check if input files are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <sincere_data.json> <cheating_data.json>"
    exit 1
fi

SINCERE_DATA=$1
CHEATING_DATA=$2
OUTPUT_DIR="keystroke_comparison_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the model
echo "Running keystroke pattern comparison..."
python simple_keystroke_comparison.py --sincere $SINCERE_DATA --cheating $CHEATING_DATA --output-dir $OUTPUT_DIR

echo "Done! Results saved to $OUTPUT_DIR" 