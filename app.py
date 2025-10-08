# app.py

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from rag_pipeline.query_pipeline import (
    load_embedding_model,
    load_llm,
    load_tokenizer,
    query_rag_pipeline
)

from rag_pipeline.gcs_loader import (
    load_chunks,
    load_embeddings,
    load_faiss_index
)

# --- App Setup ---
app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load RAG Components ---
print("🔧 Initialising RAG pipeline...")

# Load from GCS bucket
chunks = load_chunks()
embeddings = load_embeddings()
faiss_index = load_faiss_index()

embedding_model = load_embedding_model()
llm_pipeline = load_llm(mode="cloud")  
tokenizer = load_tokenizer()

print(f"✅ Loaded {len(chunks)} chunks.")
print("✅ Components ready.")

# --- Serve Frontend ---
@app.get("/")
def serve_frontend():
    return FileResponse(ROOT_DIR / "static" / "index.html")

# --- Query Endpoint ---
@app.post("/query")
async def handle_query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse(content={"answer": "⚠️ Please provide a valid question."})

    answer = query_rag_pipeline(
        question,
        embedding_model,
        faiss_index,
        chunks,
        llm_pipeline,
        tokenizer
    )
    return JSONResponse(content={"answer": answer})
