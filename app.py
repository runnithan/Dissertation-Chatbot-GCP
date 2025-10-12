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

# ✅ Add this block here
from dotenv import load_dotenv

# Load .env locally (ignored in Cloud Run)
load_dotenv()

# Diagnostic check (you can remove after verifying)
if os.getenv("K_SERVICE"):
    print("🚀 Running in Cloud Run. Secret should come from Secret Manager.")
else:
    print("💻 Running locally. Secret should come from .env file.")

if os.getenv("GROQ_API_KEY"):
    print("✅ GROQ_API_KEY detected in environment.")
else:
    print("⚠️ GROQ_API_KEY missing! Check your .env or Secret Manager setup.")

# --- App Setup ---
app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Health check route (add this here)
@app.get("/health")
def health_check():
    return {"status": "ok"}

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
