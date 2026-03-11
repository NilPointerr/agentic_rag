from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Optional

from app.config.settings import settings

if TYPE_CHECKING:
    from pinecone import Pinecone

_pc_client: Optional["Pinecone"] = None
_index = None
_init_lock = Lock()


def _list_index_names(pc: "Pinecone") -> set[str]:
    """Return existing Pinecone index names in a version-tolerant way."""

    listed = pc.list_indexes()
    names: set[str] = set()

    if isinstance(listed, list):
        for item in listed:
            if isinstance(item, dict) and "name" in item:
                names.add(item["name"])
            elif hasattr(item, "name"):
                names.add(item.name)
        return names

    if hasattr(listed, "names"):
        return set(listed.names())

    if isinstance(listed, dict) and "indexes" in listed:
        for item in listed["indexes"]:
            if isinstance(item, dict) and "name" in item:
                names.add(item["name"])
        return names

    return names


def get_pinecone_client() -> "Pinecone":
    """Create and cache the Pinecone client."""

    global _pc_client
    if _pc_client is None:
        from pinecone import Pinecone

        _pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc_client


def initialize_index() -> None:
    """Initialize the Pinecone index if it does not already exist."""

    global _index
    with _init_lock:
        if _index is not None:
            return

        pc = get_pinecone_client()
        index_name = settings.PINECONE_INDEX_NAME
        existing_indexes = _list_index_names(pc)

        if index_name not in existing_indexes:
            from pinecone import ServerlessSpec

            pc.create_index(
                name=index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )

        _index = pc.Index(index_name)


def get_index():
    """Return initialized Pinecone index handle."""

    if _index is None:
        initialize_index()
    return _index
