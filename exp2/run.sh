#!/bin/bash

# Activate virtual environment if exists
[ -d "../.venv" ] && source ../.venv/bin/activate

# Set defaults
export MAX_BATCH_SIZE=1 # NOTE: IF TESTING BY YOURSELF, anything above 1 will defintely exceed your system memory.
export TOTAL_NODES=3
export NODE_0_LB_IP=localhost:8000
export NODE_0_IP=localhost:8001
export NODE_1_IP=localhost:8002
export NODE_2_IP=localhost:8003
export FAISS_INDEX_PATH=../faiss_index.bin
export DOCUMENTS_DIR=../documents/
echo "Starting distributed monolith with load balancer..."

# Cleanup on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Start load balancer
echo "Starting load balancer on $NODE_0_IP..."
NODE_NUMBER=0 python3 load_balancer.py &
sleep 2

# Start 3 backend pipelines
echo "Starting backend 1 (Node 0 - $NODE_0_IP)..."
NODE_NUMBER=0 python3 pipeline.py &
sleep 2

echo "Starting backend 2 (Node 1 - $NODE_1_IP)..."
NODE_NUMBER=1 python3 pipeline.py &
sleep 2

echo "Starting backend 3 (Node 2 - $NODE_2_IP)..."
NODE_NUMBER=2 python3 pipeline.py &

echo "All services started. Press Ctrl+C to stop."
wait
