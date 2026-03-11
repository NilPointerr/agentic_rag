from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import TYPE_CHECKING, Iterable, Optional

from app.config.settings import settings
from app.vectorstore.pinecone_client import get_index

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: Optional["SentenceTransformer"] = None


def get_embedding_model() -> "SentenceTransformer":
    """Create and cache the sentence-transformer embedding model."""

    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate normalized embeddings for input texts."""

    model = get_embedding_model()
    return model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def build_chunk_id(doc_id: str, chunk_idx: int, text: str) -> str:
    """Build deterministic chunk id using document id, index, and text hash."""

    digest = sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{chunk_idx}:{digest}"


def embed_and_store(chunks: Iterable[dict], namespace: str = "default") -> int:
    """Embed chunk payloads and upsert them into Pinecone.

    Expected chunk item fields:
    - text: str
    - doc_id: str
    - chunk_id: int
    - source: str
    - page: int | None
    """

    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    texts = [item["text"] for item in chunk_list]
    embeddings = embed_texts(texts)

    vectors = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for item, embedding in zip(chunk_list, embeddings):
        doc_id = item["doc_id"]
        chunk_idx = int(item["chunk_id"])
        text = item["text"]

        vectors.append(
            {
                "id": build_chunk_id(doc_id, chunk_idx, text),
                "values": embedding,
                "metadata": {
                    "text": text,
                    "source": item.get("source", "unknown"),
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "page": item.get("page"),
                    "ingested_at": now_iso,
                },
            }
        )

    get_index().upsert(vectors=vectors, namespace=namespace)
    return len(vectors)
