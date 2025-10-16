# rag_pipeline/gcs_loader.py

import gcsfs
import pickle
import numpy as np
import faiss
import os

# --- Configuration ---
GCS_BUCKET = os.getenv("GCS_BUCKET", "dissertation-chatbot-data")
PROJECT_ID = os.getenv("GCP_PROJECT", "trim-artifact-470312-d5")

# --- Reuse one GCS client globally ---
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

def load_chunks(model_version="v1.0"):
    path = f"gs://{GCS_BUCKET}/models/{model_version}/chunks.pkl"
    with fs.open(path, "rb") as f:
        return pickle.load(f)

def load_embeddings(model_version="v1.0"):
    path = f"gs://{GCS_BUCKET}/models/{model_version}/embeddings.npy"
    with fs.open(path, "rb") as f:
        return np.load(f)

def load_faiss_index(model_version="v1.0"):
    path = f"gs://{GCS_BUCKET}/models/{model_version}/faiss_index.bin"
    with fs.open(path, "rb") as f:
        return faiss.read_index_binary(f)