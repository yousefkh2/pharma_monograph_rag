#!/bin/bash
# Helper script to run evaluations with proper environment setup

# Activate virtual environment
source .venv/bin/activate

# Export all variables from .env file  
set -a
source .env
set +a

# Run the evaluation with all arguments passed through
python3 eval/run_evaluation.py "$@"