from pinecone import Pinecone, ServerlessSpec
from app.config.settings import settings
from app.utils.logger import logger

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

index_name = settings.PINECONE_INDEX_NAME

# Check if index exists
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
        dimension=settings.EMBEDDING_DIMENSION,  # 384
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
logger.info(f"Connected to Pinecone index: {index_name}")


def get_index():
    """Return the configured Pinecone index client."""
    return index


def get_index_name():
    """Return the active Pinecone index name."""
    return index_name


def describe_index_stats():
    """Return index stats for the active Pinecone index."""
    return index.describe_index_stats()
