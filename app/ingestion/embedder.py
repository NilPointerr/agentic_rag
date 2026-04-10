from sentence_transformers import SentenceTransformer
from uuid import uuid4
from app.config.settings import settings
from app.vectorstore.pinecone_client import (
    describe_index_stats,
    get_index,
    get_index_name,
)
from app.utils.logger import logger

model = SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts):
    """Encode a list of texts into normalized embedding vectors."""
    return model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def embed_and_store(chunks):
    """Embed chunk records and upsert them into the vector store."""
    if not chunks:
        logger.info("No chunks available for embedding")
        return []

    if isinstance(chunks[0], str):
        chunk_records = [{"text": chunk} for chunk in chunks]
    else:
        chunk_records = chunks

    embeddings = embed_texts([chunk["text"] for chunk in chunk_records])
    vectors = []
    active_index_name = get_index_name()

    logger.info(
        f"Preparing {len(chunk_records)} chunks for Pinecone upsert into index "
        f"'{active_index_name}'. Embedding model: '{settings.EMBEDDING_MODEL}'."
    )

    for chunk, embedding in zip(chunk_records, embeddings):
        metadata = {
            key: value
            for key, value in chunk.items()
            if key != "text" and value is not None
        }
        metadata["text"] = chunk["text"]

        vectors.append({
            "id": f"doc-{uuid4()}",
            "values": embedding,
            "metadata": metadata
        })

    upsert_response = get_index().upsert(vectors=vectors)
    logger.info(
        f"Pinecone upsert completed for index '{active_index_name}'. "
        f"Requested vectors: {len(vectors)}. Response: {upsert_response}"
    )

    try:
        index_stats = describe_index_stats()
        logger.info(
            f"Pinecone index stats after upsert for '{active_index_name}': "
            f"{index_stats}"
        )
    except Exception as exc:
        logger.warning(
            f"Unable to read Pinecone index stats for '{active_index_name}': {exc}"
        )

    return embeddings
