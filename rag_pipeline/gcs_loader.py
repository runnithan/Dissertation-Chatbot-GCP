import gcsfs
import pickle
import numpy as np
import faiss
import os

# --- Configuration ---
GCS_BUCKET = os.getenv("GCS_BUCKET", "dissertation-chatbot-data")
PROJECT_ID = os.getenv("GCP_PROJECT", "trim-artifact-470312-d5")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")

# --- Reuse one GCS client globally ---
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

def load_chunks():
    path = f"gs://{GCS_BUCKET}/models/{MODEL_VERSION}/chunks.pkl"
    print(f"🔹 Loading chunks from {path}")
    with fs.open(path, "rb") as f:
        return pickle.load(f)

def load_embeddings():
    path = f"gs://{GCS_BUCKET}/models/{MODEL_VERSION}/embeddings.npy"
    print(f"🔹 Loading embeddings from {path}")
    with fs.open(path, "rb") as f:
        return np.load(f)

def load_faiss_index():
    path = f"gs://{GCS_BUCKET}/models/{MODEL_VERSION}/faiss_index.bin"
    print(f"🔹 Loading FAISS index from {path}")
    with fs.open(path, "rb") as f:
        return faiss.read_index_binary(f)
