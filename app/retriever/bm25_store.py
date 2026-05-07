import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from app.config.settings import settings
from app.utils.logger import logger

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class BM25IndexError(RuntimeError):
    """Raised when the local BM25 corpus cannot be read or written."""


@dataclass(frozen=True)
class BM25Document:
    """A chunk record stored in the local BM25 corpus."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BM25SearchResult:
    """A BM25 result with its lexical score and original chunk metadata."""

    document: BM25Document
    score: float


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical search using lowercase alphanumeric terms."""
    return TOKEN_PATTERN.findall(text.lower())


def build_chunk_id(chunk: dict[str, Any]) -> str:
    """Build a stable chunk id from source metadata and text.

    Complexity:
        O(n), where n is the length of the selected metadata and text fields.
    """
    identity = {
        "source_file": chunk.get("source_file"),
        "source_path": chunk.get("source_path"),
        "page_number": chunk.get("page_number"),
        "chunk_index": chunk.get("chunk_index"),
        "text": chunk.get("text", ""),
    }
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _corpus_path() -> Path:
    return Path(settings.BM25_INDEX_PATH)


def _document_from_chunk(chunk: dict[str, Any]) -> BM25Document | None:
    text = str(chunk.get("text", "")).strip()
    if not text:
        return None

    chunk_id = str(chunk.get("chunk_id") or build_chunk_id(chunk))
    metadata = {
        key: value
        for key, value in chunk.items()
        if key not in {"text"} and value is not None
    }
    metadata["chunk_id"] = chunk_id
    return BM25Document(chunk_id=chunk_id, text=text, metadata=metadata)


def load_documents() -> list[BM25Document]:
    """Load BM25 corpus documents from disk.

    Complexity:
        O(d), where d is the number of stored chunk records.
    """
    path = _corpus_path()
    if not path.exists():
        return []

    documents: list[BM25Document] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    document = _document_from_chunk(row)
                except (TypeError, json.JSONDecodeError) as exc:
                    logger.warning(
                        f"Skipping invalid BM25 row {line_number} in {path}: {exc}"
                    )
                    continue

                if document is not None:
                    documents.append(document)
    except OSError as exc:
        raise BM25IndexError(f"Unable to load BM25 corpus at {path}: {exc}") from exc

    return documents


def upsert_documents(chunks: Iterable[dict[str, Any]]) -> int:
    """Upsert chunk records into the local BM25 JSONL corpus.

    Complexity:
        O(e + n), where e is existing corpus size and n is the number of input
        chunks. The full file is rewritten to keep ids unique and deterministic.
    """
    path = _corpus_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BM25IndexError(f"Unable to create BM25 corpus directory: {exc}") from exc

    documents_by_id = {document.chunk_id: document for document in load_documents()}
    inserted_or_updated = 0

    for chunk in chunks:
        document = _document_from_chunk(chunk)
        if document is None:
            continue
        documents_by_id[document.chunk_id] = document
        inserted_or_updated += 1

    try:
        with path.open("w", encoding="utf-8") as handle:
            for document in documents_by_id.values():
                row = {
                    **document.metadata,
                    "chunk_id": document.chunk_id,
                    "text": document.text,
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    except OSError as exc:
        raise BM25IndexError(f"Unable to write BM25 corpus at {path}: {exc}") from exc

    logger.info(
        f"BM25 corpus upsert complete at {path}: {inserted_or_updated} chunk(s)"
    )
    return inserted_or_updated


def search_bm25(query: str, top_k: int) -> list[BM25SearchResult]:
    """Search the local corpus with Okapi BM25.

    Complexity:
        O(d * q), where d is the number of documents and q is the number of
        unique query terms after tokenization.
    """
    if top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    documents = load_documents()
    if not documents:
        return []

    tokenized_documents = [tokenize(document.text) for document in documents]
    document_count = len(tokenized_documents)
    average_document_length = (
        sum(len(tokens) for tokens in tokenized_documents) / document_count
    )
    term_document_frequencies = Counter(
        term
        for tokens in tokenized_documents
        for term in set(tokens)
    )

    k1 = settings.BM25_K1
    b = settings.BM25_B
    query_terms = set(query_tokens)
    scored_results: list[BM25SearchResult] = []

    for document, tokens in zip(documents, tokenized_documents):
        if not tokens:
            continue

        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0

        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue

            matching_docs = term_document_frequencies[term]
            idf = math.log(1 + (document_count - matching_docs + 0.5) / (matching_docs + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * document_length / average_document_length
            )
            score += idf * frequency * (k1 + 1) / denominator

        if score > 0:
            scored_results.append(BM25SearchResult(document=document, score=score))

    return sorted(scored_results, key=lambda result: result.score, reverse=True)[:top_k]
