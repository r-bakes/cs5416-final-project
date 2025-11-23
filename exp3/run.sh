#!/bin/bash

# Distribution plan:
# node 0 -> pipeline.py entrypoint + 2 instances of 01, 2 instances of 03, 2 instances of 05
# node 1 -> 2 instances of 02, 1 instance of 04
# node 2 -> 2 instances of 02, 1 instance of 04

# Activate virtual environment if exists
[ -d "../.venv" ] && source ../.venv/bin/activate

# Set node IPs
export NODE_0_IP=${NODE_0_IP:-localhost}
export NODE_1_IP=${NODE_1_IP:-localhost}
export NODE_2_IP=${NODE_2_IP:-localhost}

# Set defaults
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1}
export FAISS_INDEX_PATH=../faiss_index.bin
export DOCUMENTS_DIR=../documents/

echo "Starting microservices architecture..."
echo "Node 0: $NODE_0_IP"
echo "Node 1: $NODE_1_IP"
echo "Node 2: $NODE_2_IP"

# Cleanup on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "Starting Node 0 services..."

# NOTE: Embedding service - 2 instances
NODE_NUMBER=0 EMBEDDING_SERVICE_PORT=8001 python3 01_embedding_service.py &
sleep 2
NODE_NUMBER=0 EMBEDDING_SERVICE_PORT=8002 python3 01_embedding_service.py &
sleep 2

# NOTE: Documents service - 2 instances
NODE_NUMBER=0 DOCUMENTS_SERVICE_PORT=8003 python3 03_documents_service.py &
sleep 2
NODE_NUMBER=0 DOCUMENTS_SERVICE_PORT=8004 python3 03_documents_service.py &
sleep 2

# NOTE: Sentiment/Safety service - 2 instances
NODE_NUMBER=0 SENTIMENT_SAFETY_SERVICE_PORT=8005 python3 05_sentiment_and_safety_service.py &
sleep 2
NODE_NUMBER=0 SENTIMENT_SAFETY_SERVICE_PORT=8006 python3 05_sentiment_and_safety_service.py &
sleep 2

echo "Starting Node 1 services..."
# NOTE: FAISS service - 2 instances
NODE_NUMBER=1 FAISS_SERVICE_PORT=8007 python3 02_faiss_search_service.py &
sleep 2
NODE_NUMBER=1 FAISS_SERVICE_PORT=8008 python3 02_faiss_search_service.py &
sleep 2

# NOTE: LLM service - 1 instance
NODE_NUMBER=1 LLM_SERVICE_PORT=8009 python3 04_llm_service.py &
sleep 2

echo "Starting Node 2 services..."
# NOTE: FAISS service - 2 instances
NODE_NUMBER=2 FAISS_SERVICE_PORT=8010 python3 02_faiss_search_service.py &
sleep 5
NODE_NUMBER=2 FAISS_SERVICE_PORT=8011 python3 02_faiss_search_service.py &
sleep 5

# NOTE: LLM service - 1 instance
NODE_NUMBER=2 LLM_SERVICE_PORT=8012 python3 04_llm_service.py &
sleep 2

# Orchestrator - points to all service instances
echo "Starting orchestrator..."
export ORCHESTRATOR_PORT=8000
export EMBEDDING_SERVICE_URL="http://$NODE_0_IP:8001,http://$NODE_0_IP:8002"
export FAISS_SERVICE_URL="http://$NODE_1_IP:8007,http://$NODE_1_IP:8008,http://$NODE_2_IP:8010,http://$NODE_2_IP:8011"
export DOCUMENTS_SERVICE_URL="http://$NODE_0_IP:8003,http://$NODE_0_IP:8004"
export LLM_SERVICE_URL="http://$NODE_1_IP:8009,http://$NODE_2_IP:8012"
export SENTIMENT_SAFETY_SERVICE_URL="http://$NODE_0_IP:8005,http://$NODE_0_IP:8006"

NODE_NUMBER=0 python3 pipeline.py &
sleep 5

echo "All services started. Press Ctrl+C to stop."
echo "Orchestrator available at http://$NODE_0_IP:8000"
wait
