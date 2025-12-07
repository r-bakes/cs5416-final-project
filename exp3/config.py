import os
import torch


MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 1))
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents/")

#use CUDA if available, unless ONLY_CPU is set
ONLY_CPU = os.environ.get("ONLY_CPU", "false").lower() == "true"
if ONLY_CPU:
    DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[CONFIG] GPU acceleration enabled: {torch.cuda.get_device_name(0)}", flush=True)
else:
    DEVICE = torch.device("cpu")
    print("[CONFIG] GPU not available, using CPU", flush=True)
CONFIG = {
    "faiss_index_path": FAISS_INDEX_PATH,
    "documents_path": DOCUMENTS_DIR,
    "faiss_dim": 768,  # You must use this dimension
    "max_tokens": 128,  # You must use this max token limit
    "retrieval_k": 10,  # You must retrieve this many documents from the FAISS index
    "truncate_length": 512,  # You must use this truncate length
}

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
SAFETY_MODEL_NAME = "unitary/toxic-bert"
