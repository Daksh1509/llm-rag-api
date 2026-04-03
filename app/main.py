import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import health, query, ingest
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.utils.config import get_settings
from app.utils.logger import setup_logger
import os
# Reduce memory usage in production
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

settings = get_settings()
logger = setup_logger("main")


# ─────────────────────────────────────────────────────────────
# LIFESPAN: Startup + Shutdown events
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs code before the first request (startup) and after shutdown.
    - Startup: Load embedding model + FAISS index into memory
    - Shutdown: Graceful cleanup (flush logs, close connections)
    """
    # ── STARTUP ──────────────────────────────────────────────
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")

    logger.info("Initializing EmbeddingService (loading model + FAISS index)...")
    embedding_service.initialize()

    logger.info("Initializing LLMService (connecting to OpenAI)...")
    llm_service.initialize()

    logger.info("All services initialized. API is ready to serve requests.")

    yield  # Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("Shutting down. Cleaning up resources...")


# ─────────────────────────────────────────────────────────────
# FASTAPI APP INSTANCE
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Production-grade LLM + RAG API using FastAPI, FAISS, and OpenAI",
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc UI
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs every request with method, path, status code, and duration.
    Catches and returns 500 errors as structured JSON.
    """
    start = time.time()
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} | {duration_ms}ms"
        )
        return response
    except Exception as e:
        logger.exception(f"Unhandled exception in middleware: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


# ─────────────────────────────────────────────────────────────
# ROUTE REGISTRATION
# ─────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)