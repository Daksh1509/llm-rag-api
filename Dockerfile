# ── Stage 1: Base ────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Stage 2: Dependencies ────────────────────────────────────
FROM base AS deps

# Install system dependencies for FAISS and PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Stage 3: Final image ─────────────────────────────────────
FROM deps AS final

# Copy app source code
COPY app/ ./app/
COPY .env.example .env

# Create required directories
RUN mkdir -p data/documents embeddings

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Uvicorn with 2 workers — scale based on CPU cores (2 * CPU + 1)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]