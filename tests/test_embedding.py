import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService


def test_ingest_and_search():
    """Integration test: ingest text and verify search works."""
    svc = EmbeddingService()
    svc.initialize()

    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    a retrieval system with a language model. The retrieval system fetches
    relevant documents, which are then passed as context to the LLM.
    This grounds the model response in factual document-specific knowledge.
    """

    chunks_added = svc.ingest_text(sample_text, source_name="test_doc")
    assert chunks_added > 0

    results = svc.search("What is RAG?", top_k=3)
    assert len(results) > 0
    assert results[0]["score"] > 0.0
    assert "text" in results[0]
    assert "source" in results[0]