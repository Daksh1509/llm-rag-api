import time
from typing import Optional

from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.cache_service import cache_service
from app.models.schemas import QueryResponse, RetrievedChunk
from app.utils.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("rag_service")


class RAGService:
    """
    Orchestrates the full RAG pipeline:
    1. Check cache (skip FAISS + LLM if already answered)
    2. Embed the query and retrieve top-k chunks from FAISS
    3. Apply retrieval gate (reject if no relevant context found)
    4. Build prompt with context and call LLM
    5. Return structured response
    """

    async def query(self, user_query: str, top_k: Optional[int] = None) -> QueryResponse:
        start_time = time.time()
        top_k = top_k or settings.top_k_results

        logger.info(f"Incoming query: '{user_query[:80]}...' | top_k={top_k}")

        # ── Step 1: Cache check ─────────────────────────────────────────────
        cached_response = cache_service.get(user_query)
        if cached_response:
            cached_response.cached = True
            cached_response.latency_ms = round((time.time() - start_time) * 1000, 2)
            logger.info("Cache HIT — returning cached response")
            return cached_response

        # ── Step 2: FAISS Retrieval ─────────────────────────────────────────
        retrieved_chunks = embedding_service.search(user_query, top_k=top_k)

        # ── Step 3: Retrieval Gate ──────────────────────────────────────────
        # Reject queries where no document is sufficiently similar.
        # This prevents the LLM from hallucinating answers with no evidence.
        if not retrieved_chunks or retrieved_chunks[0]["score"] < settings.similarity_threshold:
            top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
            logger.warning(
                f"Retrieval gate REJECTED query. Top score: {top_score:.3f} "
                f"(threshold: {settings.similarity_threshold})"
            )
            return QueryResponse(
                answer="I don't have relevant documents to answer your question confidently. "
                       "Please ingest relevant documents first.",
                query=user_query,
                sources=[],
                model_used=settings.llm_model,
                cached=False,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                refusal=True,
            )

        # ── Step 4: LLM Generation ──────────────────────────────────────────
        llm_result = llm_service.generate(
            query=user_query,
            context_chunks=retrieved_chunks,
        )

        # ── Step 5: Build Response ──────────────────────────────────────────
        sources = [
            RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
                score=chunk["score"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in retrieved_chunks
        ]

        response = QueryResponse(
            answer=llm_result["answer"],
            query=user_query,
            sources=sources,
            model_used=llm_result["model_used"],
            cached=False,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            refusal=False,
        )

        # ── Step 6: Store in cache for future identical queries ─────────────
        cache_service.set(user_query, response)

        logger.info(
            f"Query completed in {response.latency_ms}ms | "
            f"chunks={len(sources)} | top_score={retrieved_chunks[0]['score']:.3f}"
        )

        return response


rag_service = RAGService()