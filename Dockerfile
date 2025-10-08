FROM python:3.10-slim

# --- Environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies ---
# build-essential + libopenblas-dev = required for faiss-cpu
# libgomp1 = required for sentence-transformers and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    build-essential \
    libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies ---
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first for cleaner installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# --- Copy source code ---
COPY . .

# --- Expose Cloud Run port ---
EXPOSE 8080

# --- Run the FastAPI app ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
