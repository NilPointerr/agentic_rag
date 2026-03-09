import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------
    # APP CONFIG
    # -------------------------
    APP_NAME: str = "Agentic RAG API"
    ENV: str = "development"
    DEBUG: bool = False

    # -------------------------
    # PINECONE CONFIG
    # -------------------------
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "agentic-rag-index-dimension-384"

    # -------------------------
    # GROQ CONFIG
    # -------------------------
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # -------------------------
    # EMBEDDING CONFIG
    # -------------------------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384


    # -------------------------
    # RETRIEVAL CONFIG
    # -------------------------
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.65

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()