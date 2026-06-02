from io import BytesIO

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

from app.api.routes import QueryRequest, ingest_documents, query_agent
from app.config.settings import settings


def _fake_upload_open(*args, **kwargs):
    """Return a minimal writable handle for upload-file save stubs."""

    class DummyFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, chunk):
            return len(chunk)

    return DummyFile()


@pytest.fixture
def fake_request():
    """Provide a minimal Starlette request object for direct route tests."""
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


def test_query_endpoint_rejects_long_queries(fake_request):
    """Ensure overlong queries are rejected before agent execution."""
    long_query = "x" * (settings.MAX_QUERY_LENGTH + 1)

    with pytest.raises(HTTPException) as exc_info:
        query_agent(fake_request, QueryRequest(query=long_query))

    assert exc_info.value.status_code == 400
    assert "character limit" in exc_info.value.detail


@pytest.mark.anyio
async def test_ingest_endpoint_rejects_path_traversal_filename(
    monkeypatch,
    fake_request,
    tmp_path,
):
    """Ensure uploaded filenames are normalized before saving to disk."""
    captured = {}
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr(
        "app.api.routes.load_pdf_pages",
        lambda path: [{"page_number": 1, "text": "safe text"}],
    )
    monkeypatch.setattr(
        "app.api.routes.chunk_pdf_pages",
        lambda **kwargs: [{"text": "chunk", "page_number": 1}],
    )
    monkeypatch.setattr("app.api.routes.embed_and_store", lambda chunks: chunks)

    def fake_open(path, mode):
        captured["path"] = str(path)
        return _fake_upload_open()

    monkeypatch.setattr("builtins.open", fake_open)

    upload = UploadFile(filename="../../etc/passwd.pdf", file=BytesIO(b"%PDF-1.4 test"))
    response = await ingest_documents(fake_request, upload)

    assert response["chunks_created"] == 1
    assert response["chunks_embedded"] == 1
    assert response["skipped_duplicate"] is False
    assert response["document_hash"]
    assert ".." not in captured["path"]
    assert captured["path"].startswith("data/uploads/")


@pytest.mark.anyio
async def test_ingest_skips_duplicate_document_by_text_hash(monkeypatch, tmp_path):
    """Ensure repeated content is skipped even when filenames differ."""
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr("app.api.routes.embed_and_store", lambda chunks: chunks)
    monkeypatch.setattr(
        "app.api.routes.load_pdf_pages",
        lambda path: [{"page_number": 1, "text": "Same PDF content"}],
    )
    monkeypatch.setattr(
        "app.api.routes.chunk_pdf_pages",
        lambda **kwargs: [{"text": "Same PDF content", "page_number": 1}],
    )
    monkeypatch.setattr("builtins.open", _fake_upload_open)

    first = await ingest_documents(
        UploadFile(filename="first.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )
    second = await ingest_documents(
        UploadFile(filename="renamed.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )

    assert first["skipped_duplicate"] is False
    assert second["skipped_duplicate"] is True
    assert second["duplicate_reason"] == "document_hash_exists"
    assert first["document_hash"] == second["document_hash"]


@pytest.mark.anyio
async def test_ingest_skips_duplicate_document_after_text_normalization(
    monkeypatch,
    tmp_path,
):
    """Ensure casing and whitespace differences do not trigger re-ingestion."""
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr("app.api.routes.embed_and_store", lambda chunks: chunks)
    monkeypatch.setattr("builtins.open", _fake_upload_open)

    page_sets = iter(
        [
            [{"page_number": 1, "text": "Hybrid Search   Uses BM25"}],
            [{"page_number": 1, "text": " hybrid\nsearch uses bm25 "}],
        ]
    )
    chunk_sets = iter(
        [
            [{"text": "Hybrid Search Uses BM25", "page_number": 1}],
            [{"text": " hybrid search uses bm25 ", "page_number": 1}],
        ]
    )

    monkeypatch.setattr("app.api.routes.load_pdf_pages", lambda path: next(page_sets))
    monkeypatch.setattr("app.api.routes.chunk_pdf_pages", lambda **kwargs: next(chunk_sets))

    first = await ingest_documents(
        UploadFile(filename="original.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )
    second = await ingest_documents(
        UploadFile(filename="formatted-copy.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )

    assert first["skipped_duplicate"] is False
    assert second["skipped_duplicate"] is True
    assert second["duplicate_reason"] == "document_hash_exists"
    assert first["document_hash"] == second["document_hash"]


@pytest.mark.anyio
async def test_ingest_processes_same_filename_when_content_changes(
    monkeypatch,
    tmp_path,
):
    """Ensure duplicate detection is based on content, not the PDF filename."""
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr("builtins.open", _fake_upload_open)

    embedded_batches = []
    page_sets = iter(
        [
            [{"page_number": 1, "text": "Version one"}],
            [{"page_number": 1, "text": "Version two"}],
        ]
    )
    chunk_sets = iter(
        [
            [{"text": "Version one", "page_number": 1}],
            [{"text": "Version two", "page_number": 1}],
        ]
    )

    monkeypatch.setattr("app.api.routes.load_pdf_pages", lambda path: next(page_sets))
    monkeypatch.setattr("app.api.routes.chunk_pdf_pages", lambda **kwargs: next(chunk_sets))
    monkeypatch.setattr(
        "app.api.routes.embed_and_store",
        lambda chunks: embedded_batches.append([chunk["text"] for chunk in chunks]),
    )

    first = await ingest_documents(
        UploadFile(filename="report.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )
    second = await ingest_documents(
        UploadFile(filename="report.pdf", file=BytesIO(b"%PDF-1.4 updated"))
    )

    assert first["skipped_duplicate"] is False
    assert second["skipped_duplicate"] is False
    assert first["document_hash"] != second["document_hash"]
    assert embedded_batches == [["Version one"], ["Version two"]]


@pytest.mark.anyio
async def test_ingest_embeds_only_new_chunks_when_document_has_partial_overlap(
    monkeypatch,
    tmp_path,
):
    """Ensure exact duplicate chunks are skipped inside an otherwise new PDF."""
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr("builtins.open", _fake_upload_open)

    embedded_batches = []
    page_sets = iter(
        [
            [{"page_number": 1, "text": "Alpha Beta"}],
            [{"page_number": 1, "text": "Alpha Beta Gamma"}],
        ]
    )
    chunk_sets = iter(
        [
            [
                {"text": "Alpha", "page_number": 1},
                {"text": "Beta", "page_number": 1},
            ],
            [
                {"text": "Alpha", "page_number": 1},
                {"text": "Beta", "page_number": 1},
                {"text": "Gamma", "page_number": 1},
            ],
        ]
    )

    monkeypatch.setattr("app.api.routes.load_pdf_pages", lambda path: next(page_sets))
    monkeypatch.setattr("app.api.routes.chunk_pdf_pages", lambda **kwargs: next(chunk_sets))
    monkeypatch.setattr(
        "app.api.routes.embed_and_store",
        lambda chunks: embedded_batches.append([chunk["text"] for chunk in chunks]),
    )

    first = await ingest_documents(
        UploadFile(filename="baseline.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )
    second = await ingest_documents(
        UploadFile(filename="expanded.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )

    assert first["chunks_embedded"] == 2
    assert first["chunks_skipped_duplicate"] == 0
    assert second["chunks_created"] == 3
    assert second["chunks_embedded"] == 1
    assert second["chunks_skipped_duplicate"] == 2
    assert second["skipped_duplicate"] is False
    assert embedded_batches == [["Alpha", "Beta"], ["Gamma"]]


@pytest.mark.anyio
async def test_ingest_can_retry_document_after_failed_embedding(
    monkeypatch,
    tmp_path,
):
    """Ensure failed dedup reservations do not permanently block a retry."""
    monkeypatch.setattr(
        "app.ingestion.dedup_store.settings.DEDUP_REGISTRY_PATH",
        str(tmp_path / "dedup_registry.json"),
    )
    monkeypatch.setattr("builtins.open", _fake_upload_open)
    monkeypatch.setattr(
        "app.api.routes.load_pdf_pages",
        lambda path: [{"page_number": 1, "text": "Retry me"}],
    )
    monkeypatch.setattr(
        "app.api.routes.chunk_pdf_pages",
        lambda **kwargs: [{"text": "Retry me", "page_number": 1}],
    )

    attempts = {"count": 0}

    def flaky_embed(chunks):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("embedding failed")
        return chunks

    monkeypatch.setattr("app.api.routes.embed_and_store", flaky_embed)

    with pytest.raises(HTTPException) as exc_info:
        await ingest_documents(
            UploadFile(filename="retry.pdf", file=BytesIO(b"%PDF-1.4 test"))
        )

    assert exc_info.value.status_code == 500

    second = await ingest_documents(
        UploadFile(filename="retry.pdf", file=BytesIO(b"%PDF-1.4 test"))
    )

    assert attempts["count"] == 2
    assert second["chunks_embedded"] == 1
    assert second["skipped_duplicate"] is False


@pytest.mark.anyio
async def test_ingest_endpoint_rejects_scanned_pdf_without_ocr(monkeypatch):
    """Ensure image-only PDFs return a clear client-facing ingestion error."""
    monkeypatch.setattr(
        "app.api.routes.load_pdf_pages",
        lambda path: (_ for _ in ()).throw(
            __import__("app.ingestion.pdf_loader", fromlist=["ImageOnlyPdfError"]).ImageOnlyPdfError(
                "This PDF appears to be image-only or scanned. No selectable text was found. Install Tesseract OCR to ingest scanned PDFs."
            )
        ),
    )

    upload = UploadFile(filename="scan.pdf", file=BytesIO(b"%PDF-1.4 test"))

    with pytest.raises(HTTPException) as exc_info:
        await ingest_documents(upload)

    assert exc_info.value.status_code == 400
    assert "image-only or scanned" in exc_info.value.detail
