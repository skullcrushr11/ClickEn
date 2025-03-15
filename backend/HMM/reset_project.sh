#!/bin/bash

# Confirm the reset
read -p "This will delete all generated files and results. Are you sure? (y/n) " CONFIRM
if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    echo "Reset cancelled."
    exit 0
fi

# Remove generated files
echo "Removing generated files..."

# Remove CSV results
find . -name "proctoring_results*.csv" -type f -delete

# Remove evaluation results
rm -rf evaluation_results/
rm -rf cross_val_results/

# Remove Python cache files
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -type f -delete
find . -name "*.pyo" -type f -delete
find . -name "*.pyd" -type f -delete

# Remove any plots saved accidentally outside directories
find . -maxdepth 1 -name "*.png" -type f -delete

echo "Project reset completed." 