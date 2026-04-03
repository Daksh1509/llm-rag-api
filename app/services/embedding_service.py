import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from app.utils.config import get_settings
from app.utils.logger import setup_logger
from app.utils.text_splitter import chunk_text, clean_text

settings = get_settings()
logger = setup_logger("embedding_service")


class EmbeddingService:
    """
    Manages the full lifecycle of embeddings:
    - Loading/saving FAISS index from disk
    - Embedding new documents and adding to index
    - Searching the index for a query

    The index is loaded ONCE at startup and kept in memory.
    This avoids re-loading the model per request (expensive!).
    """

    def __init__(self):
        self.model: SentenceTransformer = None
        self.index: faiss.Index = None
        self.metadata: List[Dict] = []  # Maps vector ID → chunk text + source
        self._initialized = False

    def initialize(self) -> None:
        """
        Called once at app startup via FastAPI lifespan.
        Loads the SentenceTransformer model and existing FAISS index (if present).
        """
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)

        self._load_index()
        self._initialized = True
        logger.info(f"EmbeddingService ready. Chunks in index: {len(self.metadata)}")

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk if they exist."""
        index_path = settings.faiss_index_path
        meta_path = settings.metadata_path

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        else:
            # Create a new flat L2 index — good up to ~100k vectors
            # For millions of vectors, switch to faiss.IndexIVFFlat
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []
            logger.info(f"Created fresh FAISS index (dim={dim})")

    def _save_index(self) -> None:
        """Persist the FAISS index and metadata to disk after each ingestion."""
        os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info("FAISS index saved to disk")

    def ingest_text(self, text: str, source_name: str) -> int:
        """
        Takes raw text, chunks it, embeds each chunk, and stores in FAISS.

        Returns:
            Number of new chunks added
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingService not initialized. Call initialize() first.")

        # 1. Clean and chunk
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, settings.chunk_size, settings.chunk_overlap)

        if not chunks:
            logger.warning(f"No chunks produced from source: {source_name}")
            return 0

        # 2. Embed all chunks in a single batch (faster than one-by-one)
        logger.info(f"Embedding {len(chunks)} chunks from '{source_name}'")
        embeddings = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

        # FAISS requires float32 specifically
        embeddings = embeddings.astype(np.float32)

        # 3. Add to FAISS index
        start_id = len(self.metadata)
        self.index.add(embeddings)

        # 4. Store metadata — maps vector ID back to original text
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_id + i,
                "text": chunk,
                "source": source_name,
                "chunk_index": i,
            })

        # 5. Persist to disk immediately
        self._save_index()

        logger.info(f"Added {len(chunks)} chunks from '{source_name}'. Total: {len(self.metadata)}")
        return len(chunks)

    def ingest_pdf(self, file_path: str, source_name: str) -> int:
        """
        Parse a PDF file and ingest its text content.
        Uses pypdf for lightweight extraction without system dependencies.
        """
        from pypdf import PdfReader

        logger.info(f"Parsing PDF: {file_path}")
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

        return self.ingest_text(full_text, source_name)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Embed a query string and retrieve the top-k most similar chunks from FAISS.

        Returns:
            List of dicts with keys: text, source, score, chunk_index
            Score is cosine similarity converted from L2 distance (higher = better)
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingService not initialized.")

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no documents ingested yet")
            return []

        top_k = top_k or settings.top_k_results

        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        # Search FAISS — returns (distances, indices)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Convert L2 distance to a 0–1 similarity score
            # Lower L2 distance = higher similarity
            similarity_score = float(1 / (1 + dist))

            chunk_meta = self.metadata[idx]
            results.append({
                "text": chunk_meta["text"],
                "source": chunk_meta["source"],
                "score": round(similarity_score, 4),
                "chunk_index": chunk_meta["chunk_index"],
            })

        logger.debug(f"Search returned {len(results)} results for query: '{query[:50]}...'")
        return results

    @property
    def total_chunks(self) -> int:
        return len(self.metadata)


# Singleton instance — shared across all FastAPI requests
embedding_service = EmbeddingService()