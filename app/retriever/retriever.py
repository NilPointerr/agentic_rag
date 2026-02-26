from app.utils.logger import logger
from app.vectorstore.pinecone_client import index
from app.ingestion.embedder import embed_texts


def retrieve(query: str, top_k=3):
    logger.info(f"Performing vector search for query: {query}")
    query_vector = embed_texts([query])[0]

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]

    if not matches:
        return [], 0.0

    avg_score = sum(m["score"] for m in matches) / len(matches)
    texts = [m["metadata"]["text"] for m in matches]

    return texts, avg_score