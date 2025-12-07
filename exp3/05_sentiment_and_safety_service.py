from transformers import pipeline as hf_pipeline
from config import CONFIG, SENTIMENT_MODEL_NAME, SAFETY_MODEL_NAME, DEVICE
import os
from flask import Flask, request, jsonify
from typing import TypedDict
from utils import profile_with_timing
from memory_profiler import profile

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
SERVICE_PORT = int(os.environ.get("SENTIMENT_SAFETY_SERVICE_PORT", 8004))


class SentimentSafetyRequest(TypedDict):
    texts: list[str]


app = Flask(__name__)
sentiment_classifier = hf_pipeline(
    "sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=DEVICE
)
safety_classifier = hf_pipeline(
    "text-classification", model=SAFETY_MODEL_NAME, device=DEVICE
)


@profile_with_timing
@profile
def _analyze_sentiment_batch(texts: list[str]) -> list[str]:
    """Step 7: Analyze sentiment for each generated response"""
    truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
    raw_results = sentiment_classifier(truncated_texts)
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
    return sentiments


@profile_with_timing
@profile
def _filter_response_safety_batch(texts: list[str]) -> list[bool]:
    """Step 8: Filter responses for safety for each entry in the batch"""
    truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
    raw_results = safety_classifier(truncated_texts)
    toxicity_flags = []
    for result in raw_results:
        toxicity_flags.append(result["score"] > 0.5)
    return toxicity_flags


@app.route("/process", methods=["POST"])
def process():
    """Analyze sentiment and safety - chains both operations"""
    try:
        data: SentimentSafetyRequest = request.json
        texts: list[str] = data.get("texts")

        sentiments = _analyze_sentiment_batch(texts)
        toxicity_flags = _filter_response_safety_batch(texts)

        return jsonify(
            {"sentiments": sentiments, "toxicity_flags": toxicity_flags}
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "service": "sentiment_safety", "node": NODE_NUMBER}
    ), 200


def main():
    """Start the sentiment and safety service"""
    print("=" * 60, flush=True)
    print("SENTIMENT & SAFETY SERVICE", flush=True)
    print("=" * 60, flush=True)
    print(f"Node: {NODE_NUMBER}", flush=True)
    print(f"Port: {SERVICE_PORT}", flush=True)
    print(f"Sentiment Model: {SENTIMENT_MODEL_NAME}", flush=True)
    print(f"Safety Model: {SAFETY_MODEL_NAME}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    app.run(host="0.0.0.0", port=SERVICE_PORT, threaded=True)


if __name__ == "__main__":
    main()
