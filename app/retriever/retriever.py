from app.utils.logger import logger
from app.vectorstore.pinecone_client import index
from app.ingestion.embedder import embed_texts


def get_index():
    """Return the configured Pinecone index instance."""
    return index


def retrieve(query: str, top_k=3):
    """Retrieve the top matching chunks and their source metadata."""
    logger.info(f"Performing vector search for query: {query}")
    query_vector = embed_texts([query])[0]

    results = get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]

    if not matches:
        return [], 0.0

    avg_score = sum(m["score"] for m in matches) / len(matches)
    sources = []

    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")

        if not text:
            continue

        source_url = metadata.get("source_url")
        page_number = metadata.get("page_number")
        page_url = source_url

        if source_url and page_number:
            page_url = f"{source_url}#page={page_number}"

        sources.append(
            {
                "text": text,
                "score": match.get("score", 0.0),
                "source_file": metadata.get("source_file"),
                "source_path": metadata.get("source_path"),
                "source_url": source_url,
                "page_number": page_number,
                "page_url": page_url,
                "chunk_index": metadata.get("chunk_index"),
            }
        )

    return sources, avg_score
