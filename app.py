# app.py

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

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

# --- Load environment ---
load_dotenv()

# --- Diagnostics ---
if os.getenv("K_SERVICE"):
    print("🚀 Running in Cloud Run. Secrets come from Secret Manager.")
else:
    print("💻 Running locally (.env).")

if os.getenv("GROQ_API_KEY"):
    print("✅ GROQ_API_KEY detected.")
else:
    print("⚠️ GROQ_API_KEY missing!")

# --- App setup ---
app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Health check ---
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "running_in": "cloud" if os.getenv("K_SERVICE") else "local",
        "groq_key_loaded": bool(os.getenv("GROQ_API_KEY"))
    }

# --- Load RAG components eagerly ---
print("🔧 Initialising RAG pipeline...")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
GCS_BUCKET = os.getenv("GCS_BUCKET", "dissertation-chatbot-data")

print(f"🧠 Loading model version: {MODEL_VERSION}")
print(f"📦 Using bucket: {GCS_BUCKET}")

try:
    chunks = load_chunks(model_version=MODEL_VERSION)
    embeddings = load_embeddings(model_version=MODEL_VERSION)
    faiss_index = load_faiss_index(model_version=MODEL_VERSION)
    print(f"✅ Loaded {len(chunks)} chunks.")
except Exception as e:
    import traceback
    print("⚠️ Failed to load RAG data from GCS:")
    print(traceback.format_exc())
    chunks = embeddings = faiss_index = None

# Always load these locally so app still starts
embedding_model = load_embedding_model()
llm_pipeline = load_llm(mode="cloud")
tokenizer = load_tokenizer()

print("✅ Components ready (startup non-blocking).")

# --- Serve frontend ---
@app.get("/")
def serve_frontend():
    return FileResponse(ROOT_DIR / "static" / "index.html")

# --- Query endpoint ---
@app.post("/query")
async def handle_query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse(content={"answer": "⚠️ Please provide a valid question."})

    try:
        answer = query_rag_pipeline(
            question,
            embedding_model,
            faiss_index,
            chunks,
            llm_pipeline,
            tokenizer
        )
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        import traceback
        print("❌ Error in query_rag_pipeline:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"answer": "⚠️ Internal error. Please try again later."}
        )
