from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.embedding_service import embedding_service
from app.utils.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns API health status and FAISS index info.
    Use this as your Docker/Kubernetes liveness probe endpoint.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        environment=settings.app_env,
        faiss_index_loaded=embedding_service._initialized,
        total_chunks_indexed=embedding_service.total_chunks,
    )