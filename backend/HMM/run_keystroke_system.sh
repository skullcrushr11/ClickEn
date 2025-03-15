#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --normal FILE       Path to sincere dataset (default: normal.json)"
    echo "  --cheating FILE      Path to cheating dataset (default: cheating.json)"
    echo "  --test-split FLOAT   Percentage of data to use for testing (default: 0.2)"
    echo "  --output DIR         Output directory for results (default: model_evaluation)"
    echo "  --by-user            Split data by user rather than by individual keystrokes"
    echo "  --optimize           Use the optimized implementation"
    echo "  --verbose            Show detailed output"
    echo "  --demo               Run in demo mode with visualization (after training)"
    echo "  --help               Show this help message"
    exit 1
}

# Check for virtual environment, create if it doesn't exist
setup_environment() {
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi
}

# Default values
SINCERE_FILE="normal.json"
CHEATING_FILE="cheating.json"
TEST_SPLIT=0.2
OUTPUT_DIR="model_evaluation"
BY_USER=false
OPTIMIZE=false
VERBOSE=false
DEMO_MODE=false

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
        --test-split)
            TEST_SPLIT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --by-user)
            BY_USER=true
            shift
            ;;
        --optimize)
            OPTIMIZE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --demo)
            DEMO_MODE=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            ;;
    esac
done

# Setup environment
setup_environment

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Split the data into train and test sets
echo "=================================================================="
echo "STEP 1: SPLITTING DATA INTO TRAIN AND TEST SETS"
echo "=================================================================="
echo "Sincere data: $SINCERE_FILE"
echo "Cheating data: $CHEATING_FILE"
echo "Test split: $TEST_SPLIT"
echo "Split by user: $BY_USER"
echo ""

SPLIT_CMD="python split_datasets.py --sincere $SINCERE_FILE --cheating $CHEATING_FILE --test-split $TEST_SPLIT --output-dir $OUTPUT_DIR"

if [ "$BY_USER" = true ]; then
    SPLIT_CMD="$SPLIT_CMD --by-user"
fi

echo "Running: $SPLIT_CMD"
SPLIT_OUTPUT=$(eval $SPLIT_CMD)

# Extract paths from the split_datasets.py output
SINCERE_TRAIN=$(echo "$SPLIT_OUTPUT" | grep "SINCERE_TRAIN:" | cut -d':' -f2)
CHEATING_TRAIN=$(echo "$SPLIT_OUTPUT" | grep "CHEATING_TRAIN:" | cut -d':' -f2)
SINCERE_TEST=$(echo "$SPLIT_OUTPUT" | grep "SINCERE_TEST:" | cut -d':' -f2)
CHEATING_TEST=$(echo "$SPLIT_OUTPUT" | grep "CHEATING_TEST:" | cut -d':' -f2)

echo ""
echo "Split complete. Generated files:"
echo "  Sincere train: $SINCERE_TRAIN"
echo "  Cheating train: $CHEATING_TRAIN"
echo "  Sincere test: $SINCERE_TEST"
echo "  Cheating test: $CHEATING_TEST"
echo ""

# Step 2: Train and evaluate the model
echo "=================================================================="
echo "STEP 2: TRAINING AND EVALUATING THE MODEL"
echo "=================================================================="

TRAIN_CMD="python train_model.py --sincere-train $SINCERE_TRAIN --cheating-train $CHEATING_TRAIN --sincere-test $SINCERE_TEST --cheating-test $CHEATING_TEST --output-dir $OUTPUT_DIR --max-samples 500"

if [ "$VERBOSE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --verbose"
fi

# Add trap for Ctrl+C
trap "echo 'Script interrupted by user. Cleaning up...'; exit 1" INT

echo "Running with a 5 minute timeout: $TRAIN_CMD"
timeout 300 $TRAIN_CMD # 300 seconds = 5 minutes

# Check if the command timed out or was interrupted
if [ $? -eq 124 ]; then
    echo "Training process timed out after 5 minutes."
    echo "Trying again with an even smaller sample size..."
    
    # Try again with a much smaller sample size
    TRAIN_CMD="$TRAIN_CMD --max-samples 100"
    timeout 300 $TRAIN_CMD
    
    if [ $? -eq 124 ]; then
        echo "Training still timed out. Using pre-trained model if available."
    fi
fi

# Extract metrics from the train_model.py output
OVERALL_ACCURACY=$(echo "$TRAIN_OUTPUT" | grep "OVERALL_ACCURACY:" | cut -d':' -f2)
SINCERE_ACCURACY=$(echo "$TRAIN_OUTPUT" | grep "SINCERE_ACCURACY:" | cut -d':' -f2)
CHEATING_ACCURACY=$(echo "$TRAIN_OUTPUT" | grep "CHEATING_ACCURACY:" | cut -d':' -f2)

# Step 3: Run demo if requested
if [ "$DEMO_MODE" = true ]; then
    echo "=================================================================="
    echo "STEP 3: RUNNING VISUALIZATION DEMO"
    echo "=================================================================="
    
    # Choose the cheating file for the demo
    DEMO_CMD="python demo.py --data $CHEATING_FILE --sincere $SINCERE_FILE --cheating $CHEATING_FILE --speed 10"
    
    echo "Running: $DEMO_CMD"
    eval $DEMO_CMD
fi

# Print final summary
echo ""
echo "=================================================================="
echo "FINAL RESULTS"
echo "=================================================================="
echo "Overall Accuracy: $OVERALL_ACCURACY"
echo "Sincere Data Accuracy: $SINCERE_ACCURACY"
echo "Cheating Data Accuracy: $CHEATING_ACCURACY"
echo ""
echo "Results have been saved to $OUTPUT_DIR"
echo "=================================================================="

# Open the output directory
if command -v xdg-open &> /dev/null; then
    xdg-open "$OUTPUT_DIR" &> /dev/null || true
elif command -v open &> /dev/null; then
    open "$OUTPUT_DIR" &> /dev/null || true
fi 