#!/bin/bash

# Black Box Challenge - Python Implementation
# This script calls the Python reimbursement calculation
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Activate virtual environment and run advanced ML model
source ml_env/bin/activate
python3 calculate_reimbursement.py "$1" "$2" "$3" 