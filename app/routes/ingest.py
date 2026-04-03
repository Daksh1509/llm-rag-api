import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import Form

from app.models.schemas import IngestResponse, IngestRequest
from app.services.embedding_service import embedding_service
from app.utils.config import get_settings
from app.utils.logger import setup_logger

router = APIRouter()
settings = get_settings()
logger = setup_logger("ingest_route")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file to be chunked, embedded, and stored in FAISS.
    Supports: .pdf, .txt, .md
    """
    # Validate extension
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file to disk
    os.makedirs(settings.documents_dir, exist_ok=True)
    file_path = os.path.join(settings.documents_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ingest based on type
    try:
        if ext.lower() == ".pdf":
            chunks_added = embedding_service.ingest_pdf(file_path, source_name=file.filename)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chunks_added = embedding_service.ingest_text(text, source_name=file.filename)
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return IngestResponse(
        message="File ingested successfully",
        source_name=file.filename,
        chunks_created=chunks_added,
        total_chunks_in_index=embedding_service.total_chunks,
    )


@router.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestRequest):
    """
    Directly ingest raw text without uploading a file.
    Useful for API-driven ingestion pipelines (e.g., from your BFA system).
    """
    try:
        chunks_added = embedding_service.ingest_text(
            text=request.text,
            source_name=request.source_name
        )
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return IngestResponse(
        message="Text ingested successfully",
        source_name=request.source_name,
        chunks_created=chunks_added,
        total_chunks_in_index=embedding_service.total_chunks,
    )