#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

# NOTE: Example execution:
# TOTAL_NODES=3 NODE_NUMBER=0 NODE_0_IP=localhost:8000 NODE_1_IP=localhost:8000 NODE_2_IP=localhost:8000 MAX_BATCH_SIZE=4 FAISS_INDEX_PATH="../faiss_index.bin" DOCUMENTS_DIR="../documents/" ./run.s

echo "Starting pipeline on Node $NODE_NUMBER..."

./exp3/run.sh
