from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.retriever.bm25_store import (
    BM25IndexError,
    build_chunk_id,
    search_bm25,
    tokenize,
)
from app.utils.logger import logger
from app.vectorstore.pinecone_client import index
from app.ingestion.embedder import embed_texts

Source = dict[str, Any]


def _preview_text(text: str, limit: int = 100) -> str:
    """Return a single-line preview for logging."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_ranked_sources(stage: str, sources: list[Source], score_key: str) -> None:
    """Log ranked source order for debugging retrieval quality."""
    if not sources:
        logger.info(f"{stage}: no sources available")
        return

    logger.info(f"{stage}: {len(sources)} source(s)")
    for idx, source in enumerate(sources, start=1):
        logger.info(
            f"{stage} #{idx} | {score_key}={source.get(score_key, 0.0):.4f} | "
            f"file={source.get('source_file')} | page={source.get('page_number')} | "
            f"chunk={source.get('chunk_index')} | text={_preview_text(source.get('text', ''))}"
        )


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


def expand_query(query: str) -> str:
    """Expand the query for lexical retrieval with normalized unique terms.

    Complexity:
        O(q), where q is the number of query tokens.
    """
    if not settings.QUERY_EXPANSION_ENABLED:
        return query

    normalized_terms = list(dict.fromkeys(tokenize(query)))
    expanded_query = " ".join([query, *normalized_terms]).strip()
    logger.info(
        f"Query expansion produced {len(normalized_terms)} lexical term(s)"
    )
    return expanded_query


def _build_source_from_metadata(metadata: dict[str, Any], score: float) -> Source | None:
    """Normalize chunk metadata into the source response shape."""
    text = str(metadata.get("text", "")).strip()
    if not text:
        return None

    source_url = metadata.get("source_url")
    page_number = metadata.get("page_number")
    page_url = source_url

    if source_url and page_number:
        page_url = f"{source_url}#page={page_number}"

    source = {
        "text": text,
        "score": float(score),
        "source_file": metadata.get("source_file"),
        "source_path": metadata.get("source_path"),
        "source_url": source_url,
        "page_number": page_number,
        "page_url": page_url,
        "chunk_index": metadata.get("chunk_index"),
    }

    if metadata.get("chunk_id"):
        source["chunk_id"] = metadata["chunk_id"]

    return source


def _source_identity(source: Source) -> str:
    """Return a stable identity for deduplicating vector and BM25 matches."""
    if source.get("chunk_id"):
        return str(source["chunk_id"])
    return build_chunk_id(source)


def _average_score(sources: list[Source], score_key: str) -> float:
    """Compute an average score safely for route quality checks."""
    if not sources:
        return 0.0
    return sum(float(source.get(score_key, 0.0)) for source in sources) / len(sources)


def _retrieve_vector_sources(query: str, candidate_k: int) -> list[Source]:
    """Retrieve semantic candidates from Pinecone.

    Complexity:
        O(e + k), where e is embedding time for the query and k is the number
        of returned vector matches.
    """
    logger.info(f"Performing vector search for query: {query}")
    query_vector = embed_texts([query])[0]

    results = get_index().query(
        vector=query_vector,
        top_k=candidate_k,
        include_metadata=True,
    )

    matches = results.get("matches", [])
    sources: list[Source] = []

    for match in matches:
        metadata = match.get("metadata", {})
        source = _build_source_from_metadata(
            metadata=metadata,
            score=float(match.get("score", 0.0)),
        )
        if source is None:
            continue
        sources.append(source)

    logger.info(
        f"Vector search returned {len(sources)} usable match(es); "
        f"candidate_k={candidate_k}"
    )
    return sources


def _retrieve_bm25_sources(query: str, candidate_k: int) -> list[Source]:
    """Retrieve lexical candidates from the local BM25 corpus.

    Complexity:
        O(d * q), where d is corpus size and q is unique query terms.
    """
    if not settings.HYBRID_SEARCH_ENABLED:
        return []

    expanded_query = expand_query(query)
    results = search_bm25(expanded_query, top_k=candidate_k)
    sources: list[Source] = []

    for result in results:
        metadata = {
            **result.document.metadata,
            "text": result.document.text,
            "chunk_id": result.document.chunk_id,
        }
        source = _build_source_from_metadata(metadata=metadata, score=result.score)
        if source is None:
            continue
        source["bm25_score"] = result.score
        sources.append(source)

    logger.info(
        f"BM25 search returned {len(sources)} usable match(es); "
        f"candidate_k={candidate_k}"
    )
    return sources


def rrf_merge(
    vector_sources: list[Source],
    bm25_sources: list[Source],
    top_k: int,
) -> list[Source]:
    """Merge vector and BM25 rankings with reciprocal rank fusion.

    Complexity:
        O(v + b + m log m), where v and b are input list sizes and m is the
        number of unique merged chunks.
    """
    if not bm25_sources:
        return vector_sources[:top_k]
    if not vector_sources:
        return bm25_sources[:top_k]

    merged: dict[str, Source] = {}
    rrf_scores: dict[str, float] = {}
    rank_constant = settings.RRF_K

    for rank, source in enumerate(vector_sources, start=1):
        identity = _source_identity(source)
        merged.setdefault(identity, source.copy())
        merged[identity]["vector_score"] = float(
            source.get("vector_score", source["score"])
        )
        rrf_scores[identity] = (
            rrf_scores.get(identity, 0.0) + 1.0 / (rank_constant + rank)
        )

    for rank, source in enumerate(bm25_sources, start=1):
        identity = _source_identity(source)
        if identity not in merged:
            merged[identity] = source.copy()
        merged[identity]["bm25_score"] = float(
            source.get("bm25_score", source["score"])
        )
        rrf_scores[identity] = (
            rrf_scores.get(identity, 0.0) + 1.0 / (rank_constant + rank)
        )

    for identity, source in merged.items():
        source["rrf_score"] = rrf_scores[identity]
        source["score"] = rrf_scores[identity]

    return sorted(
        merged.values(),
        key=lambda source: float(source["rrf_score"]),
        reverse=True,
    )[:top_k]


def rerank_sources(
    query: str,
    sources: list[Source],
    top_k: int,
) -> tuple[list[Source], float]:
    """Rerank retrieved sources with a cross-encoder when configured."""
    if not sources:
        return [], 0.0

    if not settings.RERANK_ENABLED:
        limited_sources = sources[:top_k]
        _log_ranked_sources("Retrieval ranking only", limited_sources, "score")
        return limited_sources, _average_score(limited_sources, "score")

    try:
        reranker = get_reranker()
    except Exception as exc:
        logger.warning(f"Reranker unavailable, falling back to vector scores: {exc}")
        limited_sources = sources[:top_k]
        _log_ranked_sources("Retrieval fallback ranking", limited_sources, "score")
        return limited_sources, _average_score(limited_sources, "score")

    if reranker is None:
        limited_sources = sources[:top_k]
        _log_ranked_sources("Retrieval ranking only", limited_sources, "score")
        return limited_sources, _average_score(limited_sources, "score")

    _log_ranked_sources("Before reranking", sources, "score")

    pairs = [(query, source["text"]) for source in sources]
    rerank_scores = reranker.predict(pairs)

    for source, rerank_score in zip(sources, rerank_scores):
        source["retrieval_score"] = source["score"]
        source.setdefault("vector_score", source["score"])
        source["rerank_score"] = float(rerank_score)
        source["score"] = float(rerank_score)

    reranked_sources = sorted(
        sources,
        key=lambda source: source["rerank_score"],
        reverse=True,
    )[:top_k]
    _log_ranked_sources("After reranking", reranked_sources, "rerank_score")
    return reranked_sources, _average_score(reranked_sources, "rerank_score")


def retrieve(query: str, top_k: int | None = None) -> tuple[list[Source], float]:
    """Retrieve chunks with hybrid search, RRF merge, and optional reranking.

    The pipeline is:
        query -> expansion -> BM25 + vector search -> RRF merge -> rerank.

    Complexity:
        O(e + d * q + m log m + r), where e is embedding time, d is BM25 corpus
        size, q is unique query terms, m is merged candidates, and r is reranker
        inference over the merged candidate set.
    """
    final_top_k = top_k or settings.TOP_K
    candidate_k = max(
        settings.RETRIEVAL_CANDIDATES,
        settings.RERANK_CANDIDATES if settings.RERANK_ENABLED else final_top_k,
        final_top_k,
    )

    vector_sources: list[Source] = []
    bm25_sources: list[Source] = []

    try:
        vector_sources = _retrieve_vector_sources(query, candidate_k)
    except Exception as exc:
        logger.warning(f"Vector search failed; continuing with BM25 only: {exc}")

    try:
        bm25_sources = _retrieve_bm25_sources(query, candidate_k)
    except BM25IndexError as exc:
        logger.warning(f"BM25 search unavailable; continuing with vector only: {exc}")
    except Exception as exc:
        logger.warning(f"BM25 search failed; continuing with vector only: {exc}")

    if not vector_sources and not bm25_sources:
        return [], 0.0

    merged_sources = rrf_merge(
        vector_sources=vector_sources,
        bm25_sources=bm25_sources,
        top_k=candidate_k,
    )
    _log_ranked_sources("After RRF merge", merged_sources, "score")

    return rerank_sources(query, merged_sources, final_top_k)
