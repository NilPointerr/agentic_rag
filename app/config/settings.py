from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    APP_NAME: str = "Agentic RAG API"
    ENV: str = "development"
    DEBUG: bool = False

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "agentic-rag-index-v1"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: int = 25
    LLM_MAX_RETRIES: int = 2

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.65
    MIN_RETRIEVAL_SCORE: float = 0.35
    MAX_AGENT_STEPS: int = 4

    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT_SECONDS: int = 15
    WEB_SEARCH_MAX_RETRIES: int = 1

    MAX_UPLOAD_SIZE_MB: int = 20
    RATE_LIMIT_PER_MINUTE: int = 30

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()
