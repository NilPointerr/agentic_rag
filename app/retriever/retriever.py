from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.utils.logger import logger
from app.vectorstore.pinecone_client import index
from app.ingestion.embedder import embed_texts


def get_index():
    """Return the configured Pinecone index instance."""
    return index


@lru_cache(maxsize=1)
def get_reranker():
    """Lazily load the configured reranker model."""
    logger.info("Checking if reranker is enabled...")
    if not settings.RERANK_ENABLED:
        logger.info("Reranker is not enabled.")
        return None

    logger.info(f"Loading reranker model: {settings.RERANK_MODEL}")
    return CrossEncoder(settings.RERANK_MODEL)


def rerank_sources(query: str, sources: list[dict], top_k: int) -> tuple[list[dict], float]:
    """Rerank retrieved sources with a cross-encoder when configured."""
    if not sources:
        return [], 0.0

    if not settings.RERANK_ENABLED:
        limited_sources = sources[:top_k]
        avg_score = sum(source["score"] for source in limited_sources) / len(limited_sources)
        return limited_sources, avg_score

    try:
        reranker = get_reranker()
    except Exception as exc:
        logger.warning(f"Reranker unavailable, falling back to vector scores: {exc}")
        limited_sources = sources[:top_k]
        avg_score = sum(source["score"] for source in limited_sources) / len(limited_sources)
        return limited_sources, avg_score

    if reranker is None:
        limited_sources = sources[:top_k]
        avg_score = sum(source["score"] for source in limited_sources) / len(limited_sources)
        return limited_sources, avg_score

    pairs = [(query, source["text"]) for source in sources]
    rerank_scores = reranker.predict(pairs)

    for source, rerank_score in zip(sources, rerank_scores):
        source["vector_score"] = source["score"]
        source["rerank_score"] = float(rerank_score)
        source["score"] = float(rerank_score)

    reranked_sources = sorted(
        sources,
        key=lambda source: source["rerank_score"],
        reverse=True,
    )[:top_k]
    avg_score = sum(source["rerank_score"] for source in reranked_sources) / len(reranked_sources)
    return reranked_sources, avg_score


def retrieve(query: str, top_k=3):
    """Retrieve the top matching chunks and their source metadata."""
    logger.info(f"Performing vector search for query: {query}")
    query_vector = embed_texts([query])[0]
    candidate_k = max(top_k, settings.RERANK_CANDIDATES if settings.RERANK_ENABLED else top_k)

    results = get_index().query(
        vector=query_vector,
        top_k=candidate_k,
        include_metadata=True
    )

    matches = results["matches"]

    if not matches:
        return [], 0.0

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
                "score": float(match.get("score", 0.0)),
                "source_file": metadata.get("source_file"),
                "source_path": metadata.get("source_path"),
                "source_url": source_url,
                "page_number": page_number,
                "page_url": page_url,
                "chunk_index": metadata.get("chunk_index"),
            }
        )

    return rerank_sources(query, sources, top_k)
