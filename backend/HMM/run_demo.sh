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
DATA_FILE="normal.json"
SINCERE_FILE="normal.json"
CHEATING_FILE="cheating.json"
SPEED=10
USER_ID=""
NO_PLOT=false
NO_PRETRAIN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --sincere)
            SINCERE_FILE="$2"
            shift 2
            ;;
        --cheating)
            CHEATING_FILE="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --user-id)
            USER_ID="$2"
            shift 2
            ;;
        --no-plot)
            NO_PLOT=true
            shift
            ;;
        --no-pretrain)
            NO_PRETRAIN=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Build the Python command
CMD="python demo.py --data $DATA_FILE --sincere $SINCERE_FILE --cheating $CHEATING_FILE --speed $SPEED"

# Add optional parameters
if [ ! -z "$USER_ID" ]; then
    CMD="$CMD --user-id $USER_ID"
fi

if [ "$NO_PLOT" = true ]; then
    CMD="$CMD --no-plot"
fi

if [ "$NO_PRETRAIN" = true ]; then
    CMD="$CMD --no-pretrain"
fi

# Print the command being run
echo "Running: $CMD"

# Execute the command
eval $CMD 