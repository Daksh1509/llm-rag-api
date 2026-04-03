from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Central config — reads from .env file automatically.
    All values have safe defaults so the app won't crash on startup.
    """

    # App
    app_name: str = "LLM-RAG-API"
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Embedding & RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
    similarity_threshold: float = 0.30

    # LLM Generation
    llm_model: str = "gpt-3.5-turbo"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2

    # Cache
    cache_ttl_seconds: int = 300

    # Paths
    faiss_index_path: str = "embeddings/faiss_index.bin"
    metadata_path: str = "embeddings/metadata.json"
    documents_dir: str = "data/documents"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance — same object returned on every call.
    @lru_cache ensures .env is only read once at startup, not per request.
    """
    return Settings()