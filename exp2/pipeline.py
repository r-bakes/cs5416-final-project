import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Empty, Queue
import threading

from memory_profiler import profile
import functools


# Timing decorator for profiled functions
def profile_with_timing(func):
    """Decorator that adds timing to profiled functions"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"\n[TIMING] {func_name} - START")
        start_time = time.time()

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        print(f"[TIMING] {func_name} - END (took {elapsed:.2f}s)")

        return result

    return wrapper


# Read environment variables
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 4))
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8001")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8002")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8003")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents/")

# Configuration
CONFIG = {
    "faiss_index_path": FAISS_INDEX_PATH,
    "documents_path": DOCUMENTS_DIR,
    "faiss_dim": 768,  # You must use this dimension
    "max_tokens": 128,  # You must use this max token limit
    "retrieval_k": 10,  # You must retrieve this many documents from the FAISS index
    "truncate_length": 512,  # You must use this truncate length
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()


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


class MonolithicPipeline:
    """
    Deliberately inefficient monolithic pipeline
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")

        # Model names
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"
        self.reranker_model_name = "BAAI/bge-reranker-base"
        self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.safety_model_name = "unitary/toxic-bert"

    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    @profile_with_timing
    @profile
    def process_batch(self, requests: List[PipelineRequest]) -> List[PipelineResponse]:
        """
        Main pipeline execution for a batch of requests.
        """
        if not requests:
            return []

        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]

        print("\n" + "=" * 60)
        print(f"Processing batch of {batch_size} requests")
        print("=" * 60)
        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")

        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)

        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(queries, documents_batch)

        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(queries, reranked_docs_batch)

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(
                f"\n✓ Request {request.request_id} processed in {processing_time:.2f} seconds"
            )
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(
                PipelineResponse(
                    request_id=request.request_id,
                    generated_response=responses_text[idx],
                    sentiment=sentiments[idx],
                    is_toxic=sensitivity_result,
                    processing_time=processing_time,
                )
            )

        return responses

    @profile_with_timing
    @profile
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        model = SentenceTransformer(self.embedding_model_name).to(self.device)
        embeddings = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        del model
        gc.collect()
        return embeddings

    @profile_with_timing
    @profile
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG["faiss_index_path"]):
            raise FileNotFoundError(
                "FAISS index not found. Please create the index before running the pipeline."
            )

        print("Loading FAISS index")
        index = faiss.read_index(CONFIG["faiss_index_path"])
        query_embeddings = query_embeddings.astype("float32")
        _, indices = index.search(query_embeddings, CONFIG["retrieval_k"])
        del index
        gc.collect()
        return [row.tolist() for row in indices]

    @profile_with_timing
    @profile
    def _fetch_documents_batch(
        self, doc_id_batches: List[List[int]]
    ) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                    (doc_id,),
                )
                result = cursor.fetchone()
                if result:
                    documents.append(
                        {
                            "doc_id": result[0],
                            "title": result[1],
                            "content": result[2],
                            "category": result[3],
                        }
                    )
            documents_batch.append(documents)
        conn.close()
        return documents_batch

    @profile_with_timing
    @profile
    def _rerank_documents_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        ).to(self.device)
        model.eval()
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc["content"]] for doc in documents]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=CONFIG["truncate_length"],
                ).to(self.device)
                scores = (
                    model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        del model, tokenizer
        gc.collect()
        return reranked_batches

    @profile_with_timing
    @profile
    def _generate_responses_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join(
                [f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]]
            )
            messages = [
                {
                    "role": "system",
                    "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=CONFIG["max_tokens"],
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            responses.append(response)
        del model, tokenizer
        gc.collect()
        return responses

    @profile_with_timing
    @profile
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis", model=self.sentiment_model_name, device=self.device
        )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result["label"], "neutral"))
        del classifier
        gc.collect()
        return sentiments

    @profile_with_timing
    @profile
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification", model=self.safety_model_name, device=self.device
        )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result["score"] > 0.5)
        del classifier
        gc.collect()
        return toxicity_flags


# Global pipeline instance
pipeline = None


def _process_batch_helper(batch_data):
    """Convert request data to PipelineRequest objects and process"""
    global pipeline


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
                    break  # NOTE: Fire off the request even if we don't have min batch size.

            print(f"Processing batch of {len(batch)} requests.")
            # Convert to PipelineRequest objects
            requests = [
                PipelineRequest(
                    request_id=data["request_id"],
                    query=data["query"],
                    timestamp=time.time(),
                )
                for data in batch
            ]

            # Process batch
            responses = pipeline.process_batch(requests)

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
            print(f"Error processing request: {e}")


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

        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({"request_id": request_id, "query": query})

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
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
        {"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}
    ), 200


def main():
    """
    Main execution function
    """
    global pipeline

    print("=" * 60)
    print("MONOLITHIC CUSTOMER SUPPORT PIPELINE")
    print("=" * 60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("\nNOTE: This implementation is deliberately inefficient.")
    print("Your task is to optimize this for a 3-node cluster.\n")

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()
    print("Pipeline initialized!")

    # Start worker thread
    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    print("Worker thread started!")

    # Start Flask server
    if NODE_NUMBER == 0:
        hostname, port = NODE_0_IP.split(":")
        print("Starting Flask server (Node 0)")
    elif NODE_NUMBER == 1:
        hostname, port = NODE_1_IP.split(":")
        print("Starting Flask server (Node 1)")
    else:  # NODE_NUMBER == 2
        hostname, port = NODE_2_IP.split(":")
        print("Starting Flask server (Node 2)")

    print(f"Listening on: {hostname}:{port}")
    app.run(host=hostname, port=int(port), threaded=True)


if __name__ == "__main__":
    main()
