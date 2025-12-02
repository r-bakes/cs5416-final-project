import gc
from config import DEVICE, EMBEDDING_MODEL_NAME
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from flask import Flask, request, jsonify
from typing import TypedDict
from pipeline import profile, profile_with_timing

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
SERVICE_PORT = int(os.environ.get("EMBEDDING_SERVICE_PORT", 8001))


class EmbeddingRequest(TypedDict):
    texts: list[str]


app = Flask(__name__)
model = SentenceTransformer(EMBEDDING_MODEL_NAME).to(DEVICE)

@profile_with_timing
@profile
def _generate_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Step 2: Generate embeddings for a batch of queries"""
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    gc.collect()
    return embeddings


@app.route("/process", methods=["POST"])
def process():
    """Generate embeddings for a batch of texts"""
    try:
        data: EmbeddingRequest = request.json
        texts = data.get("texts")

        embeddings = _generate_embeddings_batch(texts)

        # Convert numpy array to list for JSON serialization
        return jsonify({"embeddings": embeddings.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "service": "embedding", "node": NODE_NUMBER}
    ), 200


def main():
    """Start the embedding service"""
    print("=" * 60)
    print("EMBEDDING SERVICE")
    print("=" * 60)
    print(f"Node: {NODE_NUMBER}")
    print(f"Port: {SERVICE_PORT}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=SERVICE_PORT, threaded=True)


if __name__ == "__main__":
    main()
