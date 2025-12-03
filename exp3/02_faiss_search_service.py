from config import CONFIG, DEVICE
import numpy as np
import os
import faiss
import gc
from flask import Flask, request, jsonify
from typing import TypedDict
from utils import profile_with_timing
from memory_profiler import profile

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
SERVICE_PORT = int(os.environ.get("FAISS_SERVICE_PORT", 8002))


class FAISSRequest(TypedDict):
    embeddings: list[list[float]]


app = Flask(__name__)
index = faiss.read_index(CONFIG["faiss_index_path"])


@profile_with_timing
@profile
def _faiss_search_batch(query_embeddings: np.ndarray) -> list[list[int]]:
    """Step 3: Perform FAISS ANN search for a batch of embeddings"""
    if not os.path.exists(CONFIG["faiss_index_path"]):
        raise FileNotFoundError(
            "FAISS index not found. Please create the index before running the pipeline."
        )
    print("Loading FAISS index", flush=True)
    query_embeddings = query_embeddings.astype("float32")
    _, indices = index.search(query_embeddings, CONFIG["retrieval_k"])
    return [row.tolist() for row in indices]


@app.route("/process", methods=["POST"])
def process():
    """Perform FAISS search for batch of embeddings"""
    try:
        data: FAISSRequest = request.json
        embeddings = data.get("embeddings")

        query_embeddings = np.array(embeddings, dtype=np.float32)
        doc_ids = _faiss_search_batch(query_embeddings)

        return jsonify({"doc_ids": doc_ids}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "faiss_search",
            "node": NODE_NUMBER,
        }
    ), 200


def main():
    """Start the FAISS search service"""
    print("=" * 60, flush=True)
    print("FAISS SEARCH SERVICE", flush=True)
    print("=" * 60, flush=True)
    print(f"Node: {NODE_NUMBER}", flush=True)
    print(f"Port: {SERVICE_PORT}", flush=True)
    print("=" * 60, flush=True)

    app.run(host="0.0.0.0", port=SERVICE_PORT, threaded=True)


if __name__ == "__main__":
    main()
