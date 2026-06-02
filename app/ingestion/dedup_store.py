import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config.settings import settings

SCHEMA_VERSION = 1
_LOCK = threading.RLock()


class DeduplicationError(RuntimeError):
    """Raised when the on-disk deduplication registry cannot be accessed."""


@dataclass(frozen=True)
class DocumentDedupRecord:
    """Metadata persisted for a document-level deduplication decision."""

    document_hash: str
    uploaded_at: str
    source_filename: str
    embedding_model: str
    status: str = "indexed"


@dataclass(frozen=True)
class ChunkDedupRecord:
    """Metadata persisted for a chunk-level deduplication decision."""

    chunk_hash: str
    document_hash: str
    uploaded_at: str
    source_filename: str
    embedding_model: str
    status: str = "indexed"


@dataclass(frozen=True)
class SimilarityDedupCandidate:
    """Placeholder model for future near-duplicate or semantic dedup support."""

    content_hash: str
    embedding_model: str
    similarity_score: float | None = None


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


class DeduplicationStore:
    """Thread-safe JSON registry for exact document and chunk hashes.

    The registry is intentionally simple: it records which normalized content
    hashes have already been processed so ingestion can skip duplicate work
    before re-embedding or re-indexing the same content.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.DEDUP_REGISTRY_PATH)

    def document_exists(self, document_hash: str) -> bool:
        """Return whether the given document hash is already in the registry."""
        with _LOCK:
            data = self._load()
            return document_hash in data["documents"]

    def register_document_if_absent(self, record: DocumentDedupRecord) -> bool:
        """Register a document hash atomically before chunk processing begins.

        Returns `True` when the document was reserved for the current ingest and
        `False` when an equivalent document has already been recorded with a
        non-failed status.
        """
        with _LOCK:
            data = self._load()
            existing = data["documents"].get(record.document_hash)
            if existing and existing.get("status") != "failed":
                return False
            data["documents"][record.document_hash] = asdict(record)
            self._save(data)
            return True

    def upsert_document(self, record: DocumentDedupRecord) -> None:
        """Write the latest known status for a document hash."""
        with _LOCK:
            data = self._load()
            data["documents"][record.document_hash] = asdict(record)
            self._save(data)

    def chunk_exists(self, chunk_hash: str) -> bool:
        """Return whether the given chunk hash is already in the registry."""
        with _LOCK:
            data = self._load()
            return chunk_hash in data["chunks"]

    def filter_new_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Split chunk payloads into unseen chunks and skipped duplicate hashes.

        This helper is not used by the current route, which instead relies on
        `reserve_chunks()` for atomic reservation, but it is still useful for
        tests and offline maintenance flows.
        """
        with _LOCK:
            data = self._load()
            new_chunks: list[dict[str, Any]] = []
            skipped_hashes: list[str] = []

            for chunk in chunks:
                chunk_hash = str(chunk["chunk_hash"])
                if chunk_hash in data["chunks"]:
                    skipped_hashes.append(chunk_hash)
                    continue
                new_chunks.append(chunk)

            return new_chunks, skipped_hashes

    def reserve_chunks(self, records: list[ChunkDedupRecord]) -> set[str]:
        """Reserve unseen chunk hashes before embedding starts.

        This prevents concurrent ingest requests from embedding the same chunk at
        the same time. Returns the set of chunk hashes reserved by this caller.
        """
        if not records:
            return set()

        with _LOCK:
            data = self._load()
            reserved: set[str] = set()

            for record in records:
                existing = data["chunks"].get(record.chunk_hash)
                if existing and existing.get("status") != "failed":
                    continue
                data["chunks"][record.chunk_hash] = asdict(record)
                reserved.add(record.chunk_hash)

            self._save(data)
            return reserved

    def mark_chunks_status(self, chunk_hashes: list[str], status: str) -> None:
        """Update the stored status for previously reserved chunk hashes."""
        if not chunk_hashes:
            return

        with _LOCK:
            data = self._load()
            for chunk_hash in chunk_hashes:
                if chunk_hash in data["chunks"]:
                    data["chunks"][chunk_hash]["status"] = status
            self._save(data)

    def register_chunks(self, records: list[ChunkDedupRecord]) -> None:
        """Persist final metadata for chunk hashes that were indexed."""
        if not records:
            return

        with _LOCK:
            data = self._load()
            for record in records:
                data["chunks"][record.chunk_hash] = asdict(record)
            self._save(data)

    def _empty(self) -> dict[str, Any]:
        """Return the default registry structure for a new store."""
        return {"schema_version": SCHEMA_VERSION, "documents": {}, "chunks": {}}

    def _load(self) -> dict[str, Any]:
        """Load and normalize registry data from disk."""
        if not self.path.exists():
            return self._empty()

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DeduplicationError(
                f"Unable to load dedup registry at {self.path}: {exc}"
            ) from exc

        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("documents", {})
        data.setdefault("chunks", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        """Atomically write registry data to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=True, indent=2)
            os.replace(tmp_path, self.path)
        except OSError as exc:
            raise DeduplicationError(
                f"Unable to write dedup registry at {self.path}: {exc}"
            ) from exc
