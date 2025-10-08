# rag_pipeline/gcs_loader.py

import gcsfs
import pickle
import numpy as np
import faiss
import os

# --- Get bucket name from environment variable ---
GCS_BUCKET = os.getenv("GCS_BUCKET", "dissertation-chatbot-data")

# --- Initialise a single GCS filesystem client ---
fs = gcsfs.GCSFileSystem(project="trim-artifact-470312-d5")

def load_chunks():
    """Load document chunks from GCS."""
    print("📦 Loading chunks from GCS...")
    with fs.open(f"gs://{GCS_BUCKET}/chunks.pkl", "rb") as f:
        return pickle.load(f)

def load_embeddings():
    """Load embeddings from GCS."""
    print("📦 Loading embeddings from GCS...")
    with fs.open(f"gs://{GCS_BUCKET}/embeddings.npy", "rb") as f:
        return np.load(f)

def load_faiss_index():
    """Load FAISS index from GCS."""
    print("📦 Loading FAISS index from GCS...")
    with fs.open(f"gs://{GCS_BUCKET}/faiss_index.bin", "rb") as f:
        index_data = f.read()
        faiss_index = faiss.deserialize_index(np.frombuffer(index_data, dtype=np.uint8))
    return faiss_index
