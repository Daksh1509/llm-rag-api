from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ─────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming user query payload."""
    query: str = Field(..., min_length=3, max_length=2000, description="User's question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")

    model_config = {"json_schema_extra": {"example": {"query": "What is RAG?", "top_k": 5}}}


class IngestRequest(BaseModel):
    """Request to ingest raw text directly (alternative to file upload)."""
    text: str = Field(..., min_length=10, description="Raw text content to ingest")
    source_name: str = Field(..., min_length=1, description="Label for this document")

    model_config = {"json_schema_extra": {"example": {"text": "Retrieval-Augmented Generation is...", "source_name": "intro_doc"}}}


# ─────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with its metadata."""
    text: str
    source: str
    score: float = Field(..., description="Cosine similarity score (0.0 - 1.0)")
    chunk_index: int


class QueryResponse(BaseModel):
    """Structured response returned to client after a /query call."""
    answer: str
    query: str
    sources: List[RetrievedChunk]
    model_used: str
    cached: bool = False
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    refusal: bool = False  # True if retrieval gate rejected the query
    model_config = {"protected_namespaces": ()} 


class IngestResponse(BaseModel):
    """Response after successful document ingestion."""
    message: str
    source_name: str
    chunks_created: int
    total_chunks_in_index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    app_name: str
    environment: str
    faiss_index_loaded: bool
    total_chunks_indexed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardized error response shape."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)