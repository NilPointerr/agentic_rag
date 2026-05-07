from io import BytesIO

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

from app.api.routes import QueryRequest, ingest_documents, query_agent
from app.config.settings import settings
from app.retriever import bm25_store
from app.retriever import retriever
from app.search_tools import web_search


@pytest.fixture
def fake_request():
    """Provide a minimal Starlette request object for direct route tests."""
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


def test_retrieve_returns_all_matches(monkeypatch):
    """Ensure retrieval preserves all Pinecone matches without threshold filtering."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.92, "metadata": {"text": "high confidence"}},
                    {"score": 0.40, "metadata": {"text": "low confidence"}},
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)

    sources, score = retriever.retrieve("test query", top_k=2)

    assert sources == [
        {
            "text": "high confidence",
            "score": 0.92,
            "source_file": None,
            "source_path": None,
            "source_url": None,
            "page_number": None,
            "page_url": None,
            "chunk_index": None,
        },
        {
            "text": "low confidence",
            "score": 0.40,
            "source_file": None,
            "source_path": None,
            "source_url": None,
            "page_number": None,
            "page_url": None,
            "chunk_index": None,
        },
    ]
    assert score == 0.66


def test_retrieve_reranks_results(monkeypatch):
    """Ensure reranking can reorder Pinecone results before returning context."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.95, "metadata": {"text": "vector winner"}},
                    {"score": 0.70, "metadata": {"text": "rerank winner"}},
                ]
            }

    class FakeReranker:
        def predict(self, pairs):
            return [0.2, 0.9]

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "RERANK_CANDIDATES", 8)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retriever, "get_reranker", lambda: FakeReranker())

    sources, score = retriever.retrieve("test query", top_k=2)

    assert [source["text"] for source in sources] == ["rerank winner", "vector winner"]
    assert sources[0]["vector_score"] == 0.70
    assert sources[0]["rerank_score"] == 0.9
    assert score == 0.55


def test_retrieve_falls_back_when_reranker_unavailable(monkeypatch):
    """Ensure retrieval still succeeds if the reranker cannot be loaded."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.8, "metadata": {"text": "first"}},
                    {"score": 0.6, "metadata": {"text": "second"}},
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retriever, "get_reranker", lambda: (_ for _ in ()).throw(RuntimeError("missing model")))

    sources, score = retriever.retrieve("test query", top_k=2)

    assert [source["text"] for source in sources] == ["first", "second"]
    assert score == 0.7


def test_bm25_store_upserts_and_searches_chunks(monkeypatch, tmp_path):
    """Ensure the local BM25 corpus returns lexical matches with metadata."""
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "bm25_chunks.jsonl"),
    )

    chunks = [
        {
            "text": "pinecone vector search uses embeddings",
            "source_file": "vector.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "text": "okapi bm25 lexical search uses token frequency",
            "source_file": "bm25.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
    ]

    assert bm25_store.upsert_documents(chunks) == 2

    results = bm25_store.search_bm25("lexical token frequency", top_k=1)

    assert len(results) == 1
    assert results[0].document.metadata["source_file"] == "bm25.pdf"
    assert results[0].score > 0


def test_retrieve_merges_vector_and_bm25_with_rrf(monkeypatch, tmp_path):
    """Ensure hybrid retrieval deduplicates vector and BM25 candidates."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "bm25_chunks.jsonl"),
    )
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "RETRIEVAL_CANDIDATES", 20)

    shared_chunk = {
        "text": "hybrid retrieval combines bm25 and vector search",
        "source_file": "hybrid.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }
    shared_chunk["chunk_id"] = bm25_store.build_chunk_id(shared_chunk)
    bm25_store.upsert_documents([shared_chunk])

    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["top_k"] == 20
            return {
                "matches": [
                    {
                        "score": 0.88,
                        "metadata": shared_chunk,
                    },
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())

    sources, score = retriever.retrieve("bm25 vector hybrid", top_k=5)

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == shared_chunk["chunk_id"]
    assert sources[0]["vector_score"] == 0.88
    assert sources[0]["bm25_score"] > 0
    assert sources[0]["rrf_score"] == score


def test_web_search_returns_structured_results(monkeypatch):
    """Ensure web search output is normalized into title/body/href dictionaries."""
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results):
            return [
                {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
                {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
            ]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS, raising=False)

    results = web_search.web_search("agentic rag")

    assert results == [
        {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
    ]


def test_web_image_search_returns_structured_results(monkeypatch):
    """Ensure image search output is normalized into frontend-friendly fields."""
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def images(self, query, max_results):
            return [
                {
                    "title": "Result image",
                    "image": "https://img.example.com/full.jpg",
                    "thumbnail": "https://img.example.com/thumb.jpg",
                    "url": "https://example.com/page",
                    "source": "Example"
                }
            ]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS, raising=False)

    results = web_search.web_image_search("agentic rag")

    assert results == [
        {
            "title": "Result image",
            "image_url": "https://img.example.com/full.jpg",
            "thumbnail_url": "https://img.example.com/thumb.jpg",
            "source_url": "https://example.com/page",
            "source": "Example",
        }
    ]


def test_query_endpoint_rejects_long_queries(fake_request):
    """Ensure overlong queries are rejected before agent execution."""
    long_query = "x" * (settings.MAX_QUERY_LENGTH + 1)

    with pytest.raises(HTTPException) as exc_info:
        query_agent(fake_request, QueryRequest(query=long_query))

    assert exc_info.value.status_code == 400
    assert "character limit" in exc_info.value.detail


@pytest.mark.anyio
async def test_ingest_endpoint_rejects_path_traversal_filename(monkeypatch, fake_request):
    """Ensure uploaded filenames are normalized before saving to disk."""
    captured = {}

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

        class DummyFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, chunk):
                return len(chunk)

        return DummyFile()

    monkeypatch.setattr("builtins.open", fake_open)

    upload = UploadFile(filename="../../etc/passwd.pdf", file=BytesIO(b"%PDF-1.4 test"))
    response = await ingest_documents(fake_request, upload)

    assert response["chunks_created"] == 1
    assert ".." not in captured["path"]
    assert captured["path"].startswith("data/uploads/")


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


def test_auth_dependency_enforces_bearer_token_when_enabled():
    """Ensure auth rejects missing bearer tokens when the feature is enabled."""
    from app.security import verify_bearer_token

    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY

    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = "secret"

    try:
        with pytest.raises(HTTPException):
            verify_bearer_token(None)
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret
