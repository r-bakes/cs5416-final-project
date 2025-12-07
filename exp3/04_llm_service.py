import torch
from config import LLM_MODEL_NAME, CONFIG, DEVICE
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from flask import Flask, request, jsonify
from typing import TypedDict
from utils import profile_with_timing
from memory_profiler import profile

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
SERVICE_PORT = int(os.environ.get("LLM_SERVICE_PORT", 8005))


class LLMRequest(TypedDict):
    queries: list[str]
    documents_batch: list[list[dict]]


app = Flask(__name__)

# Load model once at startup
print("Loading LLM model...", flush=True)
#float16 for GPU float32 for CPU
dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=dtype,
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
print(f"LLM model loaded on {DEVICE} with dtype {dtype}!", flush=True)


@profile_with_timing
@profile
def _generate_responses_batch(
    queries: list[str], documents_batch: list[list[dict]]
) -> list[str]:
    """Step 6: Generate LLM responses for each query in the batch"""
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
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
    return responses


@app.route("/process", methods=["POST"])
def process():
    """Generate LLM responses for queries with documents"""
    try:
        data: LLMRequest = request.json
        queries = data.get("queries")
        documents_batch = data.get("documents_batch")

        responses = _generate_responses_batch(queries, documents_batch)

        return jsonify({"responses": responses}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "llm", "node": NODE_NUMBER}), 200


def main():
    """Start the LLM service"""
    print("=" * 60, flush=True)
    print("LLM SERVICE", flush=True)
    print("=" * 60, flush=True)
    print(f"Node: {NODE_NUMBER}", flush=True)
    print(f"Port: {SERVICE_PORT}", flush=True)
    print(f"Model: {LLM_MODEL_NAME}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    app.run(host="0.0.0.0", port=SERVICE_PORT, threaded=True)


if __name__ == "__main__":
    main()
