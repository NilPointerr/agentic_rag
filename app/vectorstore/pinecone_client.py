from pinecone import Pinecone, ServerlessSpec
from app.config.settings import settings
from app.utils.logger import logger

index_name = settings.PINECONE_INDEX_NAME
_pc: Pinecone | None = None
_index = None


def _get_pinecone_client() -> Pinecone:
    """Return a lazily-created Pinecone client."""
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def _ensure_index_exists() -> None:
    """Create the configured Pinecone index if it does not already exist."""
    pc = _get_pinecone_client()
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    logger.info(f"Pinecone configured index name: {index_name}")
    logger.info(f"Available Pinecone indexes: {existing_indexes}")

    if index_name not in existing_indexes:
        logger.info(
            "Pinecone index does not exist. Creating index "
            f"'{index_name}' with dimension {settings.EMBEDDING_DIMENSION}."
        )
        pc.create_index(
            name=index_name,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )


def get_index():
    """Return the configured Pinecone index client."""
    global _index
    if _index is None:
        _ensure_index_exists()
        _index = _get_pinecone_client().Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    return _index


class LazyPineconeIndex:
    """Compatibility proxy for modules that import `index` directly."""

    def __getattr__(self, name: str):
        return getattr(get_index(), name)


index = LazyPineconeIndex()


def get_index_name():
    """Return the active Pinecone index name."""
    return index_name


def describe_index_stats():
    """Return index stats for the active Pinecone index."""
    return index.describe_index_stats()
