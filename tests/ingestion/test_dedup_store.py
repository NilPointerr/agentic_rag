import json

import pytest

from app.ingestion.dedup_store import (
    ChunkDedupRecord,
    DeduplicationError,
    DeduplicationStore,
    DocumentDedupRecord,
)


def test_register_document_if_absent_and_document_exists(tmp_path):
    """Document registration should be idempotent for successful ingests."""
    store = DeduplicationStore(tmp_path / "dedup_registry.json")
    record = DocumentDedupRecord(
        document_hash="doc-1",
        uploaded_at="2026-06-02T00:00:00+00:00",
        source_filename="guide.pdf",
        embedding_model="test-model",
        status="processing",
    )

    assert store.register_document_if_absent(record) is True
    assert store.document_exists("doc-1") is True
    assert store.register_document_if_absent(record) is False


def test_failed_document_can_be_re_registered(tmp_path):
    """Failed documents should remain retryable for a later ingest."""
    store = DeduplicationStore(tmp_path / "dedup_registry.json")
    failed = DocumentDedupRecord(
        document_hash="doc-1",
        uploaded_at="2026-06-02T00:00:00+00:00",
        source_filename="guide.pdf",
        embedding_model="test-model",
        status="failed",
    )

    store.upsert_document(failed)

    assert store.register_document_if_absent(failed) is True


def test_reserve_chunks_and_mark_status(tmp_path):
    """Chunk reservations should skip indexed records but allow failed retries."""
    store = DeduplicationStore(tmp_path / "dedup_registry.json")
    indexed = ChunkDedupRecord(
        chunk_hash="chunk-1",
        document_hash="doc-1",
        uploaded_at="2026-06-02T00:00:00+00:00",
        source_filename="guide.pdf",
        embedding_model="test-model",
        status="indexed",
    )
    failed = ChunkDedupRecord(
        chunk_hash="chunk-2",
        document_hash="doc-1",
        uploaded_at="2026-06-02T00:00:00+00:00",
        source_filename="guide.pdf",
        embedding_model="test-model",
        status="failed",
    )

    store.register_chunks([indexed, failed])

    reserved = store.reserve_chunks(
        [
            indexed,
            ChunkDedupRecord(
                chunk_hash="chunk-2",
                document_hash="doc-1",
                uploaded_at="2026-06-02T00:00:00+00:00",
                source_filename="guide.pdf",
                embedding_model="test-model",
                status="processing",
            ),
            ChunkDedupRecord(
                chunk_hash="chunk-3",
                document_hash="doc-1",
                uploaded_at="2026-06-02T00:00:00+00:00",
                source_filename="guide.pdf",
                embedding_model="test-model",
                status="processing",
            ),
        ]
    )

    assert reserved == {"chunk-2", "chunk-3"}
    assert store.chunk_exists("chunk-1") is True
    store.mark_chunks_status(["chunk-2"], "indexed")

    data = json.loads((tmp_path / "dedup_registry.json").read_text(encoding="utf-8"))
    assert data["chunks"]["chunk-2"]["status"] == "indexed"


def test_filter_new_chunks_returns_unseen_chunks_and_skipped_hashes(tmp_path):
    """The store should separate unseen chunks from duplicates."""
    store = DeduplicationStore(tmp_path / "dedup_registry.json")
    store.register_chunks(
        [
            ChunkDedupRecord(
                chunk_hash="chunk-1",
                document_hash="doc-1",
                uploaded_at="2026-06-02T00:00:00+00:00",
                source_filename="guide.pdf",
                embedding_model="test-model",
                status="indexed",
            )
        ]
    )

    new_chunks, skipped = store.filter_new_chunks(
        [
            {"chunk_hash": "chunk-1", "text": "old"},
            {"chunk_hash": "chunk-2", "text": "new"},
        ]
    )

    assert new_chunks == [{"chunk_hash": "chunk-2", "text": "new"}]
    assert skipped == ["chunk-1"]


def test_load_raises_deduplication_error_for_invalid_json(tmp_path):
    """Invalid registry content should surface a dedicated store error."""
    path = tmp_path / "dedup_registry.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(DeduplicationError):
        DeduplicationStore(path).document_exists("doc-1")
