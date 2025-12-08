#!/bin/bash

# Distribution plan:
# node 0 -> pipeline.py entrypoint + 2 instances of 01, 2 instances of 03, 2 instances of 05, 1 instance of 02
# node 1 -> 1 instance of 02, 1 instance of 04
# node 2 -> 1 instance of 02, 1 instance of 04

# Change to exp3 directory
cd "$(dirname "$0")"

# Activate virtual environment if exists
[ -d "../.venv" ] && source ../.venv/bin/activate

# Force unbuffered Python output so logs capture prints immediately
export PYTHONUNBUFFERED=1

# Set node IPs (these come from environment)
export NODE_0_IP=${NODE_0_IP:-localhost}
export NODE_1_IP=${NODE_1_IP:-localhost}
export NODE_2_IP=${NODE_2_IP:-localhost}
export NODE_NUMBER=${NODE_NUMBER:-0}

# Parse NODE_0_IP, NODE_1_IP and NODE_2_IP to extract IP and port (format: IP:port)
if [[ "$NODE_0_IP" == *:* ]]; then
  NODE_0_HOST="${NODE_0_IP%:*}"
  NODE_0_BASE_PORT="${NODE_0_IP##*:}"
else
  NODE_0_HOST="$NODE_0_IP"
  NODE_0_BASE_PORT=8000
fi

# Parse NODE_1_IP and NODE_2_IP to extract just the host
if [[ "$NODE_1_IP" == *:* ]]; then
  NODE_1_HOST="${NODE_1_IP%:*}"
  NODE_1_BASE_PORT="${NODE_1_IP##*:}"
else
  NODE_1_HOST="$NODE_1_IP"
  NODE_1_BASE_PORT=8000
fi

if [[ "$NODE_2_IP" == *:* ]]; then
  NODE_2_HOST="${NODE_2_IP%:*}"
  NODE_2_BASE_PORT="${NODE_2_IP##*:}"
else
  NODE_2_HOST="$NODE_2_IP"
  NODE_2_BASE_PORT=8000
fi

# Calculate port numbers based on NODE_0_BASE_PORT
EMBEDDING_PORT_1=$((NODE_0_BASE_PORT + 1))
EMBEDDING_PORT_2=$((NODE_0_BASE_PORT + 2))
DOCUMENTS_PORT_1=$((NODE_0_BASE_PORT + 3))
DOCUMENTS_PORT_2=$((NODE_0_BASE_PORT + 4))
SENTIMENT_PORT_1=$((NODE_0_BASE_PORT + 5))
SENTIMENT_PORT_2=$((NODE_0_BASE_PORT + 6))
FAISS_PORT_1=$((NODE_1_BASE_PORT + 7))
LLM_PORT_1=$((NODE_1_BASE_PORT + 9))
FAISS_PORT_2=$((NODE_2_BASE_PORT + 8))
LLM_PORT_2=$((NODE_2_BASE_PORT + 10))
FAISS_PORT_3=$((NODE_1_BASE_PORT + 11))

# Update NODE_*_IP variables to contain only the host portion
export NODE_0_IP="$NODE_0_HOST"
export NODE_1_IP="$NODE_1_HOST"
export NODE_2_IP="$NODE_2_HOST"

# Set defaults
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1}
export FAISS_INDEX_PATH=${FAISS_INDEX_PATH:-../faiss_index.bin}
export DOCUMENTS_DIR=${DOCUMENTS_DIR:-../documents/}

echo "Starting microservices architecture on Node $NODE_NUMBER..."
echo "Node 0: $NODE_0_IP"
echo "Node 1: $NODE_1_IP"
echo "Node 2: $NODE_2_IP"

# Cleanup on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Start services based on NODE_NUMBER
if [ "$NODE_NUMBER" -eq 0 ]; then
  echo "Starting Node 0 services..."

  # NOTE: Embedding service - 2 instances
  EMBEDDING_SERVICE_PORT=$EMBEDDING_PORT_1 python3 01_embedding_service.py >>memory_profile_01.log &
  sleep 2
  EMBEDDING_SERVICE_PORT=$EMBEDDING_PORT_2 python3 01_embedding_service.py >>memory_profile_02.log &
  sleep 2

  # NOTE: Documents service - 2 instances
  DOCUMENTS_SERVICE_PORT=$DOCUMENTS_PORT_1 python3 03_documents_service.py >>memory_profile_03.log &
  sleep 2
  DOCUMENTS_SERVICE_PORT=$DOCUMENTS_PORT_2 python3 03_documents_service.py >>memory_profile_04.log &
  sleep 2

  # NOTE: Sentiment/Safety service - 2 instances
  SENTIMENT_SAFETY_SERVICE_PORT=$SENTIMENT_PORT_1 python3 05_sentiment_and_safety_service.py >>memory_profile_05.log &
  sleep 2
  SENTIMENT_SAFETY_SERVICE_PORT=$SENTIMENT_PORT_2 python3 05_sentiment_and_safety_service.py >>memory_profile_06.log &
  sleep 2

  FAISS_SERVICE_PORT=$FAISS_PORT_3 python3 02_faiss_search_service.py >>memory_profile_11.log &
  sleep 2

  # Orchestrator - points to all service instances
  echo "Starting orchestrator..."
  export ORCHESTRATOR_PORT=$NODE_0_BASE_PORT
  export EMBEDDING_SERVICE_URL="http://$NODE_0_IP:$EMBEDDING_PORT_1,http://$NODE_0_IP:$EMBEDDING_PORT_2"
  export FAISS_SERVICE_URL="http://$NODE_1_IP:$FAISS_PORT_1,http://$NODE_2_IP:$FAISS_PORT_2,http://$NODE_0_IP:$FAISS_PORT_3"
  export DOCUMENTS_SERVICE_URL="http://$NODE_0_IP:$DOCUMENTS_PORT_1,http://$NODE_0_IP:$DOCUMENTS_PORT_2"
  export LLM_SERVICE_URL="http://$NODE_1_IP:$LLM_PORT_1,http://$NODE_2_IP:$LLM_PORT_2"
  export SENTIMENT_SAFETY_SERVICE_URL="http://$NODE_0_IP:$SENTIMENT_PORT_1,http://$NODE_0_IP:$SENTIMENT_PORT_2"

  python3 -u pipeline.py >>memory_profile.log 2>&1 &
  sleep 5

  echo "Node 0 services started. Orchestrator available at http://$NODE_0_IP:$NODE_0_BASE_PORT"

elif [ "$NODE_NUMBER" -eq 1 ]; then
  echo "Starting Node 1 services..."

  # NOTE: FAISS service - 1 instance
  FAISS_SERVICE_PORT=$FAISS_PORT_1 python3 02_faiss_search_service.py >>memory_profile_07.log &
  sleep 2

  # NOTE: LLM service - 1 instance
  LLM_SERVICE_PORT=$LLM_PORT_1 python3 04_llm_service.py >>memory_profile_09.log &
  sleep 2

  echo "Node 1 services started."

elif [ "$NODE_NUMBER" -eq 2 ]; then
  echo "Starting Node 2 services..."

  # NOTE: FAISS service - 1 instance
  FAISS_SERVICE_PORT=$FAISS_PORT_2 python3 02_faiss_search_service.py >>memory_profile_08.log &
  sleep 2

  # NOTE: LLM service - 1 instance
  LLM_SERVICE_PORT=$LLM_PORT_2 python3 04_llm_service.py >>memory_profile_10.log &
  sleep 2

  echo "Node 2 services started."

else
  echo "ERROR: Invalid NODE_NUMBER: $NODE_NUMBER (must be 0, 1, or 2)"
  exit 1
fi

echo "All services for Node $NODE_NUMBER started. Press Ctrl+C to stop."
wait
