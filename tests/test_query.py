import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service


@pytest.fixture(autouse=True)
def initialize_services():
    """Initialize services before each test — mimics lifespan startup."""
    if not embedding_service._initialized:
        embedding_service.initialize()
    if not llm_service.client:
        llm_service.initialize()


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_chunks_indexed" in data


@pytest.mark.asyncio
async def test_query_validation():
    """Query shorter than 3 chars should fail Pydantic validation."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={"query": "hi"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_text():
    """Ingest raw text and verify chunks created."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/ingest/text", json={
            "text": "Python is a high level programming language used in AI and ML.",
            "source_name": "test_source"
        })
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] > 0