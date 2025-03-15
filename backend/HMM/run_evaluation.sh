#!/bin/bash

# Check if virtual environment exists, create if it doesn't
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Default values for parameters
SINCERE_FILE="normal.json"
CHEATING_FILE="cheating.json"
OUTPUT_DIR="evaluation_results"
USER_IDS=""
NO_PRETRAIN=false
CROSS_VALIDATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sincere)
            SINCERE_FILE="$2"
            shift 2
            ;;
        --cheating)
            CHEATING_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --user-ids)
            USER_IDS="$2"
            shift 2
            ;;
        --no-pretrain)
            NO_PRETRAIN=true
            shift
            ;;
        --cross-validate)
            CROSS_VALIDATE=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Build the Python command
CMD="python evaluate.py --sincere $SINCERE_FILE --cheating $CHEATING_FILE --output $OUTPUT_DIR"

# Add optional parameters
if [ ! -z "$USER_IDS" ]; then
    CMD="$CMD --user-ids $USER_IDS"
fi

if [ "$NO_PRETRAIN" = true ]; then
    CMD="$CMD --no-pretrain"
fi

if [ "$CROSS_VALIDATE" = true ]; then
    CMD="$CMD --cross-validate"
fi

# Print the command being run
echo "Running: $CMD"

# Execute the command
eval $CMD 