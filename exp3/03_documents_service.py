import sqlite3
from config import CONFIG, RERANKER_MODEL_NAME, DEVICE
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import os
from flask import Flask, request, jsonify
from typing import TypedDict
from pipeline import profile, profile_with_timing

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
SERVICE_PORT = int(os.environ.get("DOCUMENTS_SERVICE_PORT", 8003))


class DocumentsRequest(TypedDict):
    queries: list[str]
    doc_id_batches: list[list[int]]


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(
    DEVICE
)
model.eval()
db_path = f"{CONFIG['documents_path']}/documents.db"

@profile_with_timing
@profile
def _fetch_documents_batch(doc_id_batches: list[list[int]]) -> list[list[dict]]:
    """Step 4: Fetch documents for each query in the batch using SQLite"""
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
    queries: list[str], documents_batch: list[list[dict]]
) -> list[list[dict]]:
    """Step 5: Rerank retrieved documents for each query in the batch"""
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
            ).to(DEVICE)
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
    return reranked_batches


@app.route("/process", methods=["POST"])
def process():
    """Fetch and rerank documents - chains both operations"""
    try:
        data: DocumentsRequest = request.json
        queries = data.get("queries")
        doc_id_batches = data.get("doc_id_batches")

        # Step 1: Fetch documents
        documents_batch = _fetch_documents_batch(doc_id_batches)

        # Step 2: Rerank documents
        reranked_batch = _rerank_documents_batch(queries, documents_batch)

        return jsonify({"reranked_documents": reranked_batch}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "documents",
            "node": NODE_NUMBER,
            "db_path": f"{CONFIG['documents_path']}/documents.db",
        }
    ), 200


def main():
    """Start the documents service"""
    print("=" * 60)
    print("DOCUMENTS SERVICE (Fetch + Rerank)")
    print("=" * 60)
    print(f"Node: {NODE_NUMBER}")
    print(f"Port: {SERVICE_PORT}")
    print(f"DB: {CONFIG['documents_path']}/documents.db")
    print(f"Reranker: {RERANKER_MODEL_NAME}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=SERVICE_PORT, threaded=True)


if __name__ == "__main__":
    main()
