from sentence_transformers import SentenceTransformer
from uuid import uuid4
from app.vectorstore.pinecone_client import index
from app.utils.logger import logger

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def embed_and_store(chunks):
    if not chunks:
        logger.info("No chunks available for embedding")
        return []

    if isinstance(chunks[0], str):
        chunk_records = [{"text": chunk} for chunk in chunks]
    else:
        chunk_records = chunks

    embeddings = embed_texts([chunk["text"] for chunk in chunk_records])
    vectors = []

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

    index.upsert(vectors=vectors)
    return embeddings
