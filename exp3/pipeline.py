"""
Centralized Orchestrator for Microservices Architecture
Routes requests through the pipeline of services
"""

from config import MAX_BATCH_SIZE
import os
import time
import requests
from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
from dataclasses import dataclass

from memory_profiler import profile
import functools
import sys


# Timing decorator for profiled functions
def profile_with_timing(func):
    """Decorator that adds timing to profiled functions"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"\n[TIMING] {func_name} - START")
        sys.stdout.flush()
        start_time = time.time()

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        print(f"[TIMING] {func_name} - END (took {elapsed:.2f}s)")
        sys.stdout.flush()

        return result

    return wrapper

# Read environment variables
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", 8000))

# Service URLs - comma-separated list for multiple instances
EMBEDDING_SERVICE_URLS = os.environ.get(
    "EMBEDDING_SERVICE_URL", "http://localhost:8001"
).split(",")
FAISS_SERVICE_URLS = os.environ.get("FAISS_SERVICE_URL", "http://localhost:8002").split(
    ","
)
DOCUMENTS_SERVICE_URLS = os.environ.get(
    "DOCUMENTS_SERVICE_URL", "http://localhost:8003"
).split(",")
SENTIMENT_SAFETY_SERVICE_URLS = os.environ.get(
    "SENTIMENT_SAFETY_SERVICE_URL", "http://localhost:8004"
).split(",")
LLM_SERVICE_URLS = os.environ.get("LLM_SERVICE_URL", "http://localhost:8005").split(",")

app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()

# Round-robin counters for load balancing
lb_counters = {
    "embedding": 0,
    "faiss": 0,
    "documents": 0,
    "llm": 0,
    "sentiment_safety": 0,
}
lb_locks = {key: threading.Lock() for key in lb_counters.keys()}


def get_service_url(service_name: str, urls: list[str]) -> str:
    """Get next service URL using round-robin load balancing"""
    with lb_locks[service_name]:
        index = lb_counters[service_name] % len(urls)
        lb_counters[service_name] = index + 1
        return urls[index]


@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float


@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float

@profile_with_timing
@profile
def process_pipeline(reqs: list[PipelineRequest]) -> list[PipelineResponse]:
    """
    Orchestrate the full pipeline through microservices
    """

    batch_size = len(reqs)
    start_times = [time.time() for _ in reqs]
    queries = [req.query for req in reqs]

    print("\n" + "=" * 60, flush=True)
    print(f"Processing batch of {batch_size} requests", flush=True)
    print("=" * 60, flush=True)
    for req in reqs:
        print(f"- {req.request_id}: {req.query[:50]}...", flush=True)

    try:
        # Step 1: Generate embeddings
        embedding_url = get_service_url("embedding", EMBEDDING_SERVICE_URLS)
        print(f"[Step 1/5] Calling embedding service at {embedding_url}...", flush=True)
        response = requests.post(
            f"{embedding_url}/process", json={"texts": queries}, timeout=120
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]

        # Step 2: FAISS search
        faiss_url = get_service_url("faiss", FAISS_SERVICE_URLS)
        print(f"[Step 2/5] Calling FAISS service at {faiss_url}...", flush=True)
        response = requests.post(
            f"{faiss_url}/process", json={"embeddings": embeddings}, timeout=120
        )
        response.raise_for_status()
        doc_ids = response.json()["doc_ids"]

        # Step 3: Fetch and rerank documents
        documents_url = get_service_url("documents", DOCUMENTS_SERVICE_URLS)
        print(f"[Step 3/5] Calling documents service at {documents_url}...", flush=True)
        response = requests.post(
            f"{documents_url}/process",
            json={"queries": queries, "doc_id_batches": doc_ids},
            timeout=120,
        )
        response.raise_for_status()
        reranked_documents = response.json()["reranked_documents"]

        # Step 4: Generate LLM response
        llm_url = get_service_url("llm", LLM_SERVICE_URLS)
        print(f"[Step 4/5] Calling LLM service at {llm_url}...", flush=True)
        response = requests.post(
            f"{llm_url}/process",
            json={"queries": queries, "documents_batch": reranked_documents},
            timeout=120,
        )
        response.raise_for_status()
        llm_responses = response.json()["responses"]

        # Step 5: Sentiment and safety analysis
        sentiment_url = get_service_url(
            "sentiment_safety", SENTIMENT_SAFETY_SERVICE_URLS
        )
        print(f"[Step 5/5] Calling sentiment/safety service at {sentiment_url}...", flush=True)
        response = requests.post(
            f"{sentiment_url}/process",
            json={"texts": llm_responses},
            timeout=120,
        )
        response.raise_for_status()
        analysis = response.json()
        sentiments = analysis["sentiments"]
        toxicity_flags = analysis["toxicity_flags"]

        responses = []
        for idx, req in enumerate(reqs):
            processing_time = time.time() - start_times[idx]
            print(
                f"Request {req.request_id} processed in {processing_time:.2f} seconds",
                flush=True,
            )
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(
                PipelineResponse(
                    request_id=req.request_id,
                    generated_response=llm_responses[idx],
                    sentiment=sentiments[idx],
                    is_toxic=sensitivity_result,
                    processing_time=processing_time,
                )
            )

        return responses

    except requests.exceptions.RequestException as e:
        print(f"Batch processing failed: {e}", flush=True)
        raise
    except Exception as e:
        print(f"Batch processing error: {e}", flush=True)
        raise


def process_requests_worker():
    """Worker with adaptive batching"""
    MAX_TIMEOUT = 0.1  # 100ms under high load

    while True:
        try:
            batch = []

            # Get first request
            first_request = request_queue.get()
            if first_request is None:
                break
            batch.append(first_request)

            for _ in range(MAX_BATCH_SIZE - 1):
                try:
                    batch.append(request_queue.get(timeout=MAX_TIMEOUT))
                except Empty:
                    break  # NOTE: Fire off the request even if we don't have max batch size.

            print(f"Processing batch of {len(batch)} requests.", flush=True)

            requests = [
                PipelineRequest(
                    request_id=data["request_id"],
                    query=data["query"],
                    timestamp=time.time(),
                )
                for data in batch
            ]

            # Process batch
            responses = process_pipeline(requests)

            # Store results
            with results_lock:
                for response in responses:
                    results[response.request_id] = {
                        "request_id": response.request_id,
                        "generated_response": response.generated_response,
                        "sentiment": response.sentiment,
                        "is_toxic": response.is_toxic,
                    }
        except Exception as e:
            print(f"Error processing request: {e}", flush=True)


@app.route("/query", methods=["POST"])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get("request_id")
        query = data.get("query")

        if not request_id or not query:
            return jsonify({"error": "Missing request_id or query"}), 400

        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200

        print(f"[Orchestrator] Queueing request {request_id}", flush=True)
        # Add to queue
        request_queue.put({"request_id": request_id, "query": query})

        # Wait for processing (with timeout)
        timeout = 600  # 10 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    if "error" in result:
                        return jsonify(result), 500
                    return jsonify(result), 200

            if time.time() - start_wait > timeout:
                return jsonify({"error": "Request timeout"}), 504

            time.sleep(0.1)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "role": "orchestrator",
            "node": NODE_NUMBER,
            "services": {
                "embedding": EMBEDDING_SERVICE_URLS,
                "faiss": FAISS_SERVICE_URLS,
                "documents": DOCUMENTS_SERVICE_URLS,
                "llm": LLM_SERVICE_URLS,
                "sentiment_safety": SENTIMENT_SAFETY_SERVICE_URLS,
            },
        }
    ), 200


def main():
    """
    Main execution function
    """
    print("=" * 60, flush=True)
    print("MICROSERVICES ORCHESTRATOR", flush=True)
    print("=" * 60, flush=True)
    print(f"Orchestrator Node: {NODE_NUMBER}", flush=True)
    print(f"Port: {ORCHESTRATOR_PORT}", flush=True)
    print("\nService URLs:", flush=True)
    print(
        f"\tEmbedding ({len(EMBEDDING_SERVICE_URLS)} instances): {EMBEDDING_SERVICE_URLS}",
        flush=True,
    )
    print(f"\tFAISS ({len(FAISS_SERVICE_URLS)} instances): {FAISS_SERVICE_URLS}", flush=True)
    print(
        f"\tDocuments ({len(DOCUMENTS_SERVICE_URLS)} instances): {DOCUMENTS_SERVICE_URLS}",
        flush=True,
    )
    print(f"\tLLM ({len(LLM_SERVICE_URLS)} instances): {LLM_SERVICE_URLS}", flush=True)
    print(
        f"\tSentiment/Safety ({len(SENTIMENT_SAFETY_SERVICE_URLS)} instances): {SENTIMENT_SAFETY_SERVICE_URLS}",
        flush=True,
    )
    print("=" * 60, flush=True)

    # Start worker thread
    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    print("Worker thread started!", flush=True)

    # Start Flask server
    print(f"\nStarting Flask orchestrator on 0.0.0.0:{ORCHESTRATOR_PORT}", flush=True)
    app.run(host="0.0.0.0", port=ORCHESTRATOR_PORT, threaded=True)


if __name__ == "__main__":
    main()
