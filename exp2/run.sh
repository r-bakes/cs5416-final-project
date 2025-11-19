#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

echo "Starting pipeline on Node $NODE_NUMBER..."
source .venv/bin/activate
python3 pipeline.py >>memory_profile.log
