from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import rag_service
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger("query_route")


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Main RAG query endpoint.
    1. Embeds the user's question
    2. Retrieves relevant chunks from FAISS
    3. Passes context to LLM
    4. Returns structured answer with sources
    """
    try:
        response = await rag_service.query(
            user_query=request.query,
            top_k=request.top_k,
        )
        return response

    except RuntimeError as e:
        # Known operational errors (uninitialized services, etc.)
        logger.error(f"Runtime error in /query: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        # Unexpected errors — log full trace
        logger.exception(f"Unexpected error in /query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Check logs.")