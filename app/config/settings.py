import os
from dotenv import load_dotenv

load_dotenv(".env")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Agentic RAG API")
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = _get_bool("DEBUG", False)

    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index-dimension-384")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = _get_int("EMBEDDING_DIMENSION", 384)

    TOP_K: int = _get_int("TOP_K", 3)
    SIMILARITY_THRESHOLD: float = _get_float("SIMILARITY_THRESHOLD", 0.60)

    MCP_SERVER_URL: str | None = os.getenv("MCP_SERVER_URL","http://127.0.0.1:8000/mcp")


settings = Settings()
