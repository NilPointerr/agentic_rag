from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.ingestion.embedder import embed_texts
from app.utils.logger import logger
from app.vectorstore.pinecone_client import get_index


def retrieve(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    namespace: str = "default",
) -> dict[str, Any]:
    """Retrieve relevant passages and confidence stats for a user query."""

    effective_top_k = top_k or settings.TOP_K
    effective_min_score = (
        settings.MIN_RETRIEVAL_SCORE if min_score is None else float(min_score)
    )

    logger.info("Performing vector search")
    query_vector = embed_texts([query])[0]

    results = get_index().query(
        vector=query_vector,
        top_k=effective_top_k,
        include_metadata=True,
        namespace=namespace,
    )

    matches = results.get("matches", [])
    passages: list[dict[str, Any]] = []

    for match in matches:
        score = float(match.get("score", 0.0))
        if score < effective_min_score:
            continue

        metadata = match.get("metadata") or {}
        passages.append(
            {
                "text": metadata.get("text", ""),
                "score": score,
                "source": metadata.get("source", "unknown"),
                "doc_id": metadata.get("doc_id", "unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "page": metadata.get("page"),
            }
        )

    top_score = passages[0]["score"] if passages else 0.0
    avg_score = (
        sum(item["score"] for item in passages) / len(passages) if passages else 0.0
    )

    return {
        "passages": passages,
        "top_score": top_score,
        "avg_score": avg_score,
        "used_top_k": effective_top_k,
        "used_min_score": effective_min_score,
    }
