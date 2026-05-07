from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------
    # APP CONFIG
    # -------------------------
    APP_NAME: str = "Agentic RAG API"
    ENV: str = "development"
    DEBUG: bool = False
    MAX_QUERY_LENGTH: int = 20000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    AUTH_ENABLED: bool = False
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"

    # -------------------------
    # PINECONE CONFIG
    # -------------------------
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "agentic-rag-index-v2"

    # -------------------------
    # GROQ CONFIG
    # -------------------------
    GROQ_API_KEY: str
    GROQ_MODEL: str = "mixtral-8x7b-32768"

    # -------------------------
    # EMBEDDING CONFIG
    # -------------------------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384


    # -------------------------
    # RETRIEVAL CONFIG
    # -------------------------
    TOP_K: int = 5
    RETRIEVAL_CANDIDATES: int = 20
    HYBRID_SEARCH_ENABLED: bool = True
    QUERY_EXPANSION_ENABLED: bool = True
    RRF_K: int = 60
    BM25_INDEX_PATH: str = "data/bm25_chunks.jsonl"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings():
    """Return a cached settings object loaded from environment variables."""
    return Settings()


settings = get_settings()
