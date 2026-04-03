from typing import List
from app.utils.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("text_splitter")


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Splits a long string into overlapping fixed-size chunks.

    Why overlapping chunks?
    - A sentence split at a boundary loses context.
    - Overlap ensures the answer isn't lost at a chunk boundary.

    Args:
        text: Raw document text
        chunk_size: Max characters per chunk (default from config)
        chunk_overlap: Overlap between consecutive chunks (default from config)

    Returns:
        List of text chunk strings
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:  # Skip empty chunks
            chunks.append(chunk)

        # Move start forward, but step back by overlap to maintain continuity
        start += chunk_size - chunk_overlap

    logger.debug(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove control characters from extracted PDF text.
    """
    import re
    # Replace multiple newlines/tabs with single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()